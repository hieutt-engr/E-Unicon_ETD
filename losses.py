from __future__ import print_function
import torch

"""
Author: Aiyang Han (aiyangh@nuaa.edu.cn)
Date: May 24th, 2022
"""

import torch.nn as nn
import torch.nn.functional as F


class UniConLoss_Standard(nn.Module):
    """Universum-inspired Supervised Contrastive Learning: https://arxiv.org/abs/2204.10695"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(UniConLoss_Standard, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, universum, labels):
        """
        We include universum data into the calculation of InfoNCE.
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            universum: universum data of shape [bsz*n_views, ...]
        Returns:
            A loss scalar.
        """
        # Get device from `features`
        device = features.device

        # Check and synchronize device for tensors
        labels = labels.to(device)
        universum = universum.to(device)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # Synchronize device for `mask`
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # shape of [bsz*n_views, feature_dimension]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # only show one view, shape of [bsz, feature_dimension]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # show all the views, shape of [bsz*n_views, feature_dimension]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, universum.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # find the biggest
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # make the size suited for similarity matrix

        # mask-out self-contrast cases, make value on the diagonal False
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class UniConLoss_ETD(nn.Module):
    """
    Supervised Contrastive loss + universum negatives.
    - features: [B, D] or [B, V, D]
    - labels:   [B]
    - universum: [U, D] or [B, D] or [B*V, D]  (optional; if None -> pure SupCon)
    """
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.1, eps=1e-8, l2norm=True):
        super().__init__()
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = float(base_temperature)
        self.eps = eps
        self.l2norm = l2norm

    @staticmethod
    def _l2(x, dim=-1, eps=1e-6):
        return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)

    def forward(self, features, labels, universum=None):
        """
        NOTE: Đổi thứ tự cho dễ dùng: (features, labels, universum)
        - Nếu bạn muốn giữ chữ ký (features, universum, labels) thì gọi bằng keyword.
        """
        if labels is None:
            raise ValueError("labels is required")
        device = features.device

        # Canonicalize features -> [B, V, D]
        if features.dim() == 2:
            B, D = features.shape
            V = 1
            feats = features.view(B, 1, D)
        elif features.dim() == 3:
            B, V, D = features.shape
            feats = features
        else:
            raise ValueError(f"`features` must be [B,D] or [B,V,D], got {tuple(features.shape)}")

        if self.l2norm:
            feats = self._l2(feats, dim=2)

        labels = labels.to(device).long().view(-1)
        if labels.shape[0] != B:
            raise ValueError("Num of labels does not match num of features")

        # Positives mask by class (base [B,B])
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)  # [B,B]

        # Build SupCon anchors & contrasts FROM features
        contrast_feature = torch.cat(torch.unbind(feats, dim=1), dim=0)  # [B*V, D]
        if self.contrast_mode == 'one':
            anchor_feature = feats[:, 0]  # [B, D]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [B*V, D]
            anchor_count = V
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Build denominator bank = [contrast_feature; universum?]
        if universum is not None:
            if universum.dim() != 2 or universum.size(1) != contrast_feature.size(1):
                raise ValueError("universum must be [N,D] with same D as features")
            U = universum.size(0)
            uni = universum.to(device)
            if self.l2norm:
                uni = self._l2(uni, dim=1)
            bank = torch.cat([contrast_feature, uni], dim=0)  # [B*V + U, D]
            has_uni = True
        else:
            bank = contrast_feature                                  # [B*V, D]
            has_uni = False
            U = 0

        # Compute logits: anchors vs bank
        logits = torch.matmul(anchor_feature, bank.t()) / self.temperature  # [B*V, B*V + U] or [B,B] if V=1,no uni

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Build mask over positives ONLY in the first B*V columns (i.e., contrast_feature part)
        mask_pos = mask.repeat(anchor_count, V)  # [B*V, B*V]
        if has_uni:
            # append zeros for universum part
            zeros = torch.zeros(mask_pos.size(0), U, device=device, dtype=mask_pos.dtype)
            mask_full = torch.cat([mask_pos, zeros], dim=1)  # [B*V, B*V+U]
        else:
            mask_full = mask_pos  # [B*V, B*V]

        # Mask-out self-contrast on SupCon part (diagonal)
        logits_mask = torch.ones_like(mask_full, device=device)
        if self.contrast_mode == 'all':
            diag_len = B * V
        else:  # 'one'
            diag_len = B
        logits_mask.scatter_(1, torch.arange(diag_len, device=device).view(-1,1), 0.0)
        mask_full = mask_full * logits_mask

        # Log-softmax over bank
        exp_logits = torch.exp(logits) * logits_mask
        denom = exp_logits.sum(1, keepdim=True).clamp_min(self.eps)
        log_prob = logits - torch.log(denom)

        # Average log-likelihood over positives
        pos_cnt = mask_full.sum(1).clamp_min(self.eps)
        mean_log_prob_pos = (mask_full * log_prob).sum(1) / pos_cnt

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, B).mean()
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

def vicreg_loss(z1, z2, sim_w=25.0, var_w=25.0, cov_w=1.0, eps=1e-4):
    inv = F.mse_loss(z1, z2)

    def _var(z):
        std = z.std(dim=0) + eps
        return torch.mean(F.relu(1.0 - std))

    var = _var(z1) + _var(z2)

    z1c, z2c = z1 - z1.mean(0), z2 - z2.mean(0)
    N = z1.size(0)
    cov1 = (z1c.T @ z1c) / (N - 1)
    cov2 = (z2c.T @ z2c) / (N - 1)
    cov = off_diagonal(cov1).pow_(2).mean() + off_diagonal(cov2).pow_(2).mean()

    return sim_w*inv + var_w*var + cov_w*cov

def recon_loss(x_hat, x, w_l1=1.0):
    return w_l1 * F.l1_loss(x_hat, x)
