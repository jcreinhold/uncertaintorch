#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
loss

holds loss func definitions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['GaussianDiagLoss',
           'L1OnlyLoss',
           'LaplacianDiagLoss',
           'MSEOnlyLoss',
           'FocalLoss',
           'ExtendedCrossEntropy']

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MaskLoss(nn.Module):
    def __init__(self, beta=25.):
        super().__init__()
        self.beta = beta

    def forward(self, out, y):
        if y.size(1) == 1:
            loss = self.loss_fn(out, y)
        else:
            tgt, mask = torch.chunk(y, 2, dim=1)
            mask *= self.beta
            mask[mask == 0.] = 1.
            mask /= self.beta
            loss = torch.mean(mask * self.loss_fn(out, tgt, reduction='none'))
        return loss

    def loss_fn(self, out, y, reduction='mean'):
        raise NotImplementedError


class MSEOnlyLoss(MaskLoss):
    def loss_fn(self, out, y, reduction='mean'):
        yhat, _ = out
        return F.mse_loss(yhat, y, reduction=reduction)


class L1OnlyLoss(MaskLoss):
    def loss_fn(self, out, y, reduction='mean'):
        yhat, _ = out
        return F.l1_loss(yhat, y, reduction=reduction)


class GaussianDiagLoss(MaskLoss):
    def loss_fn(self, out, y, reduction='mean'):
        yhat, s = out
        loss = 0.5 * (torch.exp(-s) * F.mse_loss(yhat, y, reduction='none') + s)
        if reduction == 'mean': loss = torch.mean(loss)
        return loss


class LaplacianDiagLoss(MaskLoss):
    def loss_fn(self, out, y, reduction='mean'):
        yhat, s = out
        loss = np.sqrt(2) * (torch.exp(-s) * F.l1_loss(yhat, y, reduction='none')) + s
        if reduction == 'mean': loss = torch.mean(loss)
        return loss


class MaskLossSegmentation(nn.Module):
    def __init__(self, beta=25., use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.beta = beta

    def forward(self, out, y):
        if not self.use_mask:
            loss = self.loss_fn(out, y)
        else:
            tgt, mask = torch.chunk(y, 2, dim=1)
            tgt, mask = tgt.squeeze(), mask.squeeze().float()
            mask *= self.beta
            mask[mask == 0.] = 1.
            mask /= self.beta
            loss = torch.mean(mask * self.loss_fn(out, tgt, reduction='none'))
        return loss

    def loss_fn(self, out, y, reduction='mean'):
        raise NotImplementedError


class FocalLoss(MaskLossSegmentation):
    def __init__(self, beta=25., use_mask=False, weight=None, gamma=2.):
        super().__init__(beta, use_mask)
        self.weight = weight
        self.gamma = gamma

    def loss_fn(self, out, y, reduction='mean'):
        pred, _ = out
        log_prob = F.log_softmax(pred, dim=1)
        prob = torch.exp(log_prob)
        p = ((1 - prob) ** self.gamma) * log_prob
        return F.nll_loss(p, y, weight=self.weight, reduction=reduction)


class ExtendedCrossEntropy(MaskLossSegmentation):
    def __init__(self, beta=25., use_mask=False, weight=None, n_samples=10):
        super().__init__(beta, use_mask)
        self.weight = weight
        self.nsamp = n_samples

    def loss_fn(self, out, y, reduction='mean'):
        logits, sigma = out
        dist = torch.distributions.Normal(logits, sigma)
        x_hat = dist.rsample((self.nsamp,))
        mc_prob = F.softmax(x_hat, dim=2).mean(dim=0)  # channel dim = 2 b/c samples
        return F.nll_loss(mc_prob.log(), y, weight=self.weight, reduction=reduction)

