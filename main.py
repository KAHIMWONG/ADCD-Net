# <editor-fold desc="header">
import os

import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus

import torch

import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

torch.set_num_threads(1)

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


torch.Tensor.__repr__ = tensor_info

import os.path as op
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.model import ADCDNet
from loss.soft_ce_loss import SoftCrossEntropyLoss
from loss.lovasz_loss import LovaszLoss
from ds import get_train_dl, get_val_dl, get_general_val_dl
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


# </editor-fold>

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.tb_writer = SummaryWriter(cfg.tb_log)
        self.ckpt_dir = op.join(cfg.exp_dir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = ADCDNet()
        if cfg.ckpt is not None:
            self.load_ckpt(cfg.ckpt)
        self.model = nn.DataParallel(self.model).cuda()
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, cfg.epochs, eta_min=cfg.min_lr)
        self.ce = SoftCrossEntropyLoss(smooth_factor=0.1, reduction="mean", ignore_index=None)
        self.lovasz = LovaszLoss(mode='multiclass', per_image=True)
        self.l1 = nn.L1Loss()
        self.align_ce = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

    def train(self):
        self.train_dl = get_train_dl()
        step, min_qf = 1, 100
        self.model.train()

        for epoch in range(1, cfg.epochs + 1):
            losses_record = defaultdict(AverageMeter)

            if epoch != 1:
                self.train_dl.dataset.S += cfg.cnt_per_epoch

            for items in self.train_dl:
                img, dct, qt, mask, ocr_mask, is_align, min_qf = \
                    (
                        items['img'].cuda(),
                        items['dct'].cuda(),
                        items['qt'].cuda().unsqueeze(1),
                        items['mask'].cuda(),
                        items['ocr_mask'].cuda(),
                        items['is_align'].cuda(),
                        items['min_qf'][0]
                    )

                with autocast():
                    logits, loc_feat, align_logits, rec_output, focal_losses = self.model(img, dct, qt, mask, ocr_mask,
                                                                                          is_train=True)

                # ------------------ LOSS ------------------

                # reconstruction loss
                rec_img, shuffle_rec_img, norm_dct = rec_output
                img_l1_loss = self.l1(rec_img[:, :3], img) + self.l1(shuffle_rec_img[:, :3], img)
                dct_l1_loss = self.l1(rec_img[:, -1], norm_dct) + self.l1(shuffle_rec_img[:, -1], norm_dct)
                rec_loss = cfg.rec_w * (img_l1_loss + dct_l1_loss)

                # dct align score loss
                align_loss = self.align_ce(align_logits, is_align.long())

                # focal loss
                focal_loss = [cfg.focal_w[idx] * (loss.sum() / (loss != 0).sum()) for idx, loss in
                              enumerate(focal_losses)]
                focal_loss = torch.stack(focal_loss).sum()

                # for training stability
                norm_loss = cfg.norm_w * loc_feat.norm(dim=1).mean()

                ce_loss = cfg.ce_w * self.ce(logits.float(), mask)
                iou_loss = self.lovasz(logits.float(), mask)
                total_loss = ce_loss + iou_loss + focal_loss + rec_loss + align_loss + norm_loss

                with torch.no_grad():
                    f1, p, r = self.compute_f1(logits, mask)
                    align_acc = (align_logits.argmax(1) == is_align).float().mean().item()

                losses = {
                    'total': total_loss.item(),
                    'ce': ce_loss.item(),
                    'iou': iou_loss.item(),
                    'rec': rec_loss.item(),
                    'align_ce': align_loss.item(),
                    'focal': focal_loss.item(),
                    'norm': norm_loss.item(),
                    'f1': f1,
                    'align_acc': align_acc
                }

                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                for name, loss in losses.items():
                    losses_record[name].update(loss)

                if cfg.check_val:
                    _ = self.val()
                    self.save_ckpt(epoch, 0)

                if step % cfg.print_log_step == 0:
                    self.print_log(step, losses_record, min_qf)
                    self.write_tb(step, losses_record)

                step += 1

            if epoch % cfg.val_epoch == 0:
                val_score = self.val()
                self.save_ckpt(epoch, val_score)
                self.print_log(step, losses_record, min_qf)
                logging.info('Score: %5.4f' % val_score)

            self.scheduler.step()

    def val(self):
        self.val_dls = get_val_dl()
        self.model.eval()
        with torch.no_grad():
            val_f1_list = []
            for val_name, dl in self.val_dls.items():
                logging.info('Val Set: %s' % val_name)
                p_list, r_list = [], []
                for items in tqdm(dl):
                    img, dct, qt, mask, ocr_mask, img_name, ori_img = \
                        (
                            items['img'].cuda(),
                            items['dct'].cuda(),
                            items['qt'].cuda().unsqueeze(1),
                            items['mask'].cuda(),
                            items['ocr_mask'].cuda(),
                            items['img_name'],
                            items['ori_img']
                        )

                    with autocast():
                        logits = self.model(img, dct, qt, mask, ocr_mask, is_train=False)[0]

                    per_f1, per_p, per_r = self.compute_f1(logits, mask)
                    p_list.append(per_p)
                    r_list.append(per_r)

                p = np.array(p_list).mean()
                r = np.array(r_list).mean()
                f1 = (2 * p * r / (p + r + 1e-8))
                logging.info('Precision:%5.4f Recall:%5.4f F1:%5.4f' % (p, r, f1))
                val_f1_list.append(f1)

        f1 = np.array(val_f1_list).mean()
        self.model.train()
        return f1

    def general_val(self):
        self.gen_val_dls = get_general_val_dl()
        self.model.eval()
        with torch.no_grad():
            val_f1_list = []
            for val_name, dl in self.gen_val_dls.items():
                logging.info('Val Set: %s' % val_name)
                p_list, r_list = [], []
                for items in tqdm(dl):
                    img, dct, qt, mask, ocr_mask, img_names, sizes = \
                        (
                            items[0].cuda(),
                            items[1].cuda(),
                            items[2].cuda().unsqueeze(1),
                            items[3].cuda(),
                            items[4].cuda(),
                            items[5],
                            items[6]
                        )

                    with autocast():
                        logits = self.model(img, dct, qt, mask, ocr_mask, is_train=False)[0]

                    for logit, each_y, (h, w), name in zip(logits, mask, sizes, img_names):
                        if name != 'padding':
                            crop_logit = logit[..., :h, :w].unsqueeze(0)
                            crop_y = each_y[..., :h, :w].unsqueeze(0)
                            per_f1, per_p, per_r = self.compute_f1(crop_logit, crop_y)
                            p_list.append(per_p)
                            r_list.append(per_r)

                    # per_f1, per_p, per_r = self.compute_f1(logits, y)
                    # p_list.append(per_p)
                    # r_list.append(per_r)

                    # if cfg.save_vis:
                    #     self.save_output(img_name, seg_gt, ori_img, [F.softmax(logits, dim=1)[:, 1]])

                p = np.array(p_list).mean()
                r = np.array(r_list).mean()
                f1 = (2 * p * r / (p + r + 1e-8))
                logging.info('Precision:%5.4f Recall:%5.4f F1:%5.4f' % (p, r, f1))
                val_f1_list.append(f1)

        f1 = np.array(val_f1_list).mean()
        self.model.train()
        return f1

    def write_tb(self, cnt, losses_record):
        for loss_name, loss_value in losses_record.items():
            self.tb_writer.add_scalar('losses/{}'.format(loss_name.strip()), loss_value.val,
                                      global_step=cnt)

    def print_log(self, step, losses_record, min_qf):
        lr = self.optimizer.param_groups[0]['lr']

        output = 'Step (%6d/%6d): lr:%.2e min_qf:%3d' % (step, cfg.total_step, lr, min_qf)
        for name, loss in losses_record.items():
            output += ' %s:%5.4f' % (name, loss.val)
        logging.info(output)

    def compute_f1(self, logit, y):
        with torch.no_grad():
            if len(logit.shape) == 3:
                pred = F.sigmoid(logit) > 0.5
            elif len(logit.shape) == 4:
                pred = logit.argmax(1)  # ori [b,h,w]
            y_ = y.squeeze(1)
            matched = (pred * y_).sum((1, 2))
            pred_sum = pred.sum((1, 2))
            y_sum = y_.sum((1, 2))
            p = (matched / (pred_sum + 1e-8)).mean().item()
            r = (matched / (y_sum + 1e-8)).mean().item()
            f1 = (2 * p * r / (p + r + 1e-8))
        return f1, p, r

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        miss, unexpect = self.model.load_state_dict(self.modify_cp_dict(ckpt['model']), strict=False)
        print('Missed keys:', miss)
        print('Unexpected keys:', unexpect)

    def modify_cp_dict(self, cp_dict):
        new_cp_dict = {}
        for key in cp_dict:
            if key.startswith('module.'):
                new_cp_dict[key[7:]] = cp_dict[key]
            else:
                new_cp_dict[key] = cp_dict[key]
        return new_cp_dict

    def save_ckpt(self, epoch, score):
        state_dict = {
            'model': self.model.state_dict(),
        }
        torch.save(state_dict, op.join(self.ckpt_dir, 'Ep_%s_%5.4f.pth' % (epoch, score)))


if __name__ == '__main__':
    trainer = Trainer()
    if cfg.mode == 'train':
        trainer.train()
    elif cfg.mode == 'val':
        trainer.val()
    elif cfg.mode == 'general_val':
        trainer.general_val()
