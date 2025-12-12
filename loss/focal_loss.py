import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def supcon_parallel(f, y, t=0.1, sample_n=512, min_n=3):
    b, c, h, w = f.shape

    # reshape y to the same size of f
    y = F.interpolate(y.float(), size=(h, w), mode='nearest').long()

    l = h * w
    f = f.permute(0, 2, 3, 1).reshape(b, l, c)
    y = y.reshape(b, l)

    # sample
    f_list, y_list, len_list = [], [], []
    for b_idx in range(b):  # iter each sample
        bf = f[b_idx]
        by = y[b_idx]
        r_f = bf[by == 0]
        f_f = bf[by == 1]
        r_n, f_n = r_f.size(0), f_f.size(0)
        if r_n < min_n or f_n < min_n:
            continue

        sample_r_f = r_f[torch.randperm(r_f.size(0))[:sample_n]]
        sample_f_f = f_f[torch.randperm(f_f.size(0))[:sample_n]]

        sample_f = torch.cat([sample_r_f, sample_f_f], 0)
        sample_y = torch.cat([torch.zeros(sample_r_f.size(0)), torch.ones(sample_f_f.size(0))], 0)

        f_list.append(sample_f)
        y_list.append(sample_y)
        len_list.append(sample_f.size(0))

    if len(f_list) == 0:
        return torch.tensor([0.0]).cuda()

    pad_f = pad_sequence(f_list, batch_first=True, padding_value=1)  # [b, max_l, c]
    y = pad_sequence(y_list, batch_first=True, padding_value=-1).cuda()  # [b, max_l]
    is_pad = pad_f.sum(-1) == c  # [b, max_l]
    f = F.normalize(pad_f, dim=-1)
    b, l, c = pad_f.shape

    # compute similarity matrix
    sim = torch.bmm(f, f.permute(0, 2, 1))
    sim = torch.exp(torch.div(sim, t))  # temp

    is_valid = ~is_pad
    valid_mask = torch.bmm(is_valid[:, :, None].float(), is_valid[:, None, :].float())

    # define pos mask
    p_mask = (y[:, None, :] == y[:, :, None]).float()
    # fill diagonal with 0
    eyes = torch.eye(l, dtype=torch.bool).cuda().repeat(b, 1, 1)
    reverse_eyes = ~eyes
    p_mask = p_mask * reverse_eyes * valid_mask
    p_num = torch.sum(p_mask, dim=-1)

    # define neg mask
    n_mask = (y[:, None, :] != y[:, :, None]).float()
    n_mask = n_mask * reverse_eyes * valid_mask

    # compute loss
    denominator_p = torch.sum(sim * p_mask, dim=-1, keepdim=True)
    denominator_n = torch.sum(sim * n_mask, dim=1, keepdim=True)
    denominator = denominator_p + denominator_n

    logits = torch.sum(torch.log(sim / (denominator + 1e-8)) * p_mask, dim=-1)

    logits = (-logits / (p_num + 1e-8))

    logits = logits.mean()

    return logits

