import torch
import torch.nn as nn
import torch.nn.functional as F
import cfg
from loss.focal_loss import *
from model.fph import FPH, AddCoord
from model.restormer import get_restormer


def get_mlp(in_channels, out_channels=None, bias=True):
    if out_channels is None:
        out_channels = in_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels // 2, 1, bias=bias),
        nn.GELU(),
        nn.Conv2d(in_channels // 2, out_channels, 1, bias=bias))


class ADCDNet(nn.Module):
    def __init__(self,
                 cls_n = 2,
                 loc_out_dim=96,  # number of classes
                 rec_out_dim=4,  # reconstruction output channels RGB + DCT
                 dct_feat_dim=256,
                 focal_in_dim=[192, 96, 96, 96],
                 focal_out_dim=32,
                 pp_scale_n=4,  # number of channels in the pp map
                 ):
        super().__init__()
        # rgb encoder + localization branch
        self.restormer_loc = get_restormer(model_name='full_model', out_channels=loc_out_dim)
        self.load_docres()
        # reconstruction branch
        self.restormer_rec = get_restormer(model_name='decoder_only', out_channels=rec_out_dim)
        # dct encoder
        self.dct_encoder = FPH()
        # alignment score predictor
        self.dct_align_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dct_feat_dim, dct_feat_dim // 2),
            nn.BatchNorm1d(dct_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(dct_feat_dim // 2, 2)
        )
        self.add_coord = AddCoord(with_r=False)

        self.focal_proj = nn.ModuleList([get_mlp(in_channels=focal_in_dim[i], out_channels=focal_out_dim) for i in range(len(focal_in_dim))])

        # pristine prototype estimation
        self.pp_scale_proj = get_mlp(in_channels=pp_scale_n, out_channels=loc_out_dim)
        self.pp_bias_proj = get_mlp(in_channels=pp_scale_n, out_channels=loc_out_dim)
        self.out_head = get_mlp(in_channels=loc_out_dim, out_channels=cls_n)


    def forward(self, img, dct, qt, mask, ocr_mask, is_train=True):
        img_size = img.size(2)

        # get alignment score & multi-scale dct features
        as_feat, ms_dct_feats = self.dct_encoder(dct, qt)
        align_logits = self.dct_align_head(as_feat)
        align_score = F.softmax(align_logits, dim=1)[:, -1]

        # get rgb feature & localization
        feat, loc_feat, cnt_feats, frg_feats, pp_feats = self.restormer_loc(img=torch.cat([self.add_coord(img), torch.zeros_like(ocr_mask)], 1),
                                                                              ms_dct_feats=ms_dct_feats,
                                                                              dct_align_score=align_score,
                                                                              ocr_mask=ocr_mask,
                                                                              img_size=img_size)

        pp_maps = self.get_pp_map(pp_feats, ocr_mask, img_size)
        pp_scale = self.pp_scale_proj(pp_maps)
        pp_bias = self.pp_bias_proj(pp_maps)
        feat = feat * pp_scale + pp_bias
        logits = self.out_head(feat)

        if not is_train:  # val
            rec_output, focal_losses = None, None
        else:  # train
            # reconstruction branch
            # rec_img = self.restormer_rec(cnt_feats, frg_feats, is_shuffle=False)
            shuffle_rec_img = self.restormer_rec(cnt_feats, frg_feats, is_shuffle=True)
            norm_dct = (dct.float() / 20.0)
            rec_output = (shuffle_rec_img, norm_dct)

            # focal
            focal_losses = tuple([supcon_parallel(self.focal_proj[i](pp_feats[i]), mask)
                                  for i in range(len(pp_feats))])

        return logits, pp_feats, align_logits, rec_output, focal_losses

    def get_pp_map(self, pp_feats, y, img_size):
        maps_per_level = []

        # pre-compute background mask once; it gets resized inside the loop
        y = y.float()

        for f in pp_feats:  # -- loop only over pyramid levels
            f = F.normalize(f, p=2, dim=1)  # [B, C, H, W], channel-wise ℓ₂-norm
            _, c, h, w = f.shape

            # resize mask to feature resolution and create 0/1 background mask
            bg_mask = (F.interpolate(y, size=(h, w), mode="nearest") == 0).float()  # [B, 1, H, W]

            # -------- background mean feature (vectorised over batch) ----------
            bg_sum = (f * bg_mask).sum(dim=(2, 3))  # [B, C]
            bg_count = bg_mask.sum(dim=(2, 3)).clamp_min(1.0)  # avoid ÷0
            bg_mean = F.normalize(bg_sum / bg_count, p=2, dim=1)  # [B, C]

            # -------- cosine-similarity map (dot product: both vectors are normed) ----------
            sim = (f * bg_mean[:, :, None, None]).sum(dim=1, keepdim=True)  # [B, 1, H, W]

            # upscale to final resolution
            sim = F.interpolate(sim, size=(img_size, img_size),
                                mode="bilinear", align_corners=False)  # [B, 1, S, S]

            maps_per_level.append(sim)  # (keep extra channel)

        # L maps stacked along channel dim → [B, L, S, S]
        return torch.cat(maps_per_level, dim=1)

    def load_docres(self):
        ckpt = torch.load(cfg.docres_ckpt_path, map_location='cpu')['model_state']
        # remove 'output' layer in ckpt
        for name in list(ckpt.keys()):
            if 'output' in name:
                ckpt.pop(name)
        miss, unexpected = self.restormer_loc.load_state_dict(self.rm_ckpt_module(ckpt), strict=False)
        print('Missed keys:', miss)
        print('Unexpected keys:', unexpected)

    def rm_ckpt_module(self, cp_dict):
        new_cp_dict = {}
        for key in cp_dict:
            if key.startswith('module.'):
                new_cp_dict[key[7:]] = cp_dict[key]
            else:
                new_cp_dict[key] = cp_dict[key]
        return new_cp_dict