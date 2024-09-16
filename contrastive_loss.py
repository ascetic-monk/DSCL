import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Intra_f_to_f(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(Intra_f_to_f, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        # self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        mask = self.mask_correlated_samples(batch_size)
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class Intra_p_to_p(nn.Module):
    def __init__(self, class_num, temperature, args, device):
        super(Intra_p_to_p, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.args = args
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss, ne_loss


def inter_p_to_f(indcls, feats, device):
    feats_1, feats_2, feats_3 = feats

    # Mask Generation
    indcls = indcls.contiguous().view(-1, 1)
    mask = torch.eq(indcls, indcls.T).float().to(device)
    # mask = (mask * conf.unsqueeze(0) * conf.unsqueeze(1)).detach()

    supcon = SupConLoss_()

    supcon_loss = (supcon(feats_1, feats_2, mask=mask) + supcon(feats_2, feats_3, mask=mask)) / 2
    return supcon_loss


#cluster-wise
class SupConLoss_(nn.Module):
    def __init__(self, temperature=1.):
        super(SupConLoss_, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels=None, mask=None, device='cuda', m=0):
        b = z1.shape[0]
        assert z1.shape[0] == z2.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(b, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != b:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        v = 2
        z = torch.cat((z1, z2), dim=0)
        # z = torch.cat(torch.unbind(z, dim=1), dim=0)

        # compute logits
        sim_z = torch.matmul(z, z.T) / self.temperature
        # for numerical stability
        sim_z_max, _ = torch.max(sim_z, dim=1, keepdim=True)
        sim_z = sim_z - sim_z_max.detach()

        # tile mask
        mask = mask.repeat(v, v)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b * v).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_sim_z = torch.exp(sim_z) * logits_mask
        log_sim = sim_z - torch.log(exp_sim_z.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive and calculate loss
        msk_sum = (mask > 0).sum(1)
        balance_weight = (1. / (msk_sum * (msk_sum + 1))) / (1. / (msk_sum + 1)).sum()
        # balance_weight = (1. / (msk_sum))/msk_sum.shape[0]
        loss = -((mask * log_sim).sum(1) * balance_weight).sum()
        return loss

def inter_f_to_p(args, feat, target, prob, prob2, prob_n, conf_msk, indcls):
    p1, pn = prob, prob_n

    labeled_len = len(target)
    feat_detach = feat.detach()

    feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
    cosine_dist = torch.mm(feat_norm, feat_norm.t())

    # 1. Generation of Positive Pairs
    # 1.1. labeled part.
    indcls = indcls.cpu().numpy()
    indcls_raw = np.zeros(conf_msk.shape[0])

    conf_msk_np = np.zeros(len(conf_msk,)).astype(np.bool)
    conf_msk_np[:labeled_len] = True
    indcls_raw[conf_msk_np] = indcls[:labeled_len]

    b = feat.shape[0]
    pos_pairs = []

    for i in range(b):
        if conf_msk_np[i]:
            target_i = indcls_raw[i]
            idxs = np.where((indcls_raw == target_i) * (conf_msk_np))[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
                if i > labeled_len: conf_msk_np[i] = 0
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
        else:
            pos_pairs.append(0)
    # print(conf_msk_np.sum())
    conf_msk_del = torch.tensor(conf_msk_np).cuda()
    p2 = conf_msk_del.detach().unsqueeze(1) * prob2[pos_pairs] + (~conf_msk_del).detach().unsqueeze(1) * pn
    # p2 = conf_msk_del.detach().unsqueeze(1) * prob2[pos_pairs] + (~conf_msk_del).detach().unsqueeze(1) * pn

    # 2. Generation of Negative Samples Thresholds.
    cos_sub = np.uint8(255 * cosine_dist[conf_msk][:, conf_msk].cpu().detach().numpy())
    thr, cos_thres = cv2.threshold(cos_sub, 0, 1, cv2.THRESH_OTSU)
    costhr = float(thr) / 255.

    neg_mask = cosine_dist < costhr
    ind_cls_cuda = torch.tensor(indcls).cuda().view(-1, 1)
    neg_mask[conf_msk, :][:, conf_msk] = ~torch.eq(ind_cls_cuda, ind_cls_cuda.T)

    # p1, p2, prob2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1), F.normalize(prob2, dim=1)
    temperature = 1.0
    b = p1.shape[0]

    pos_neg_matrix = torch.cat((p1, prob2), dim=0)
    pos_neg_sim = torch.mm(p1, pos_neg_matrix.T)/ temperature
    pos_sim_add_exp = torch.exp((p1 * p2).sum(1)/ temperature)
    pos_mask = torch.cat((torch.zeros((b, b)), torch.eye(b, b)), dim=1).cuda()
    neg_mask = torch.cat((neg_mask, neg_mask), dim=1)

    pos_neg_sim = pos_neg_sim
    pos_neg_sim_exp = torch.exp(pos_neg_sim)
    loss = -((2 * torch.log(pos_sim_add_exp / (pos_sim_add_exp * 2 + ((pos_mask + neg_mask) * pos_neg_sim_exp).sum(1))))+
             torch.log((pos_neg_sim_exp * pos_mask).sum(1) / (pos_sim_add_exp * 2 + ((pos_mask + neg_mask) * pos_neg_sim_exp).sum(1))) ).mean() / 3


    return loss

