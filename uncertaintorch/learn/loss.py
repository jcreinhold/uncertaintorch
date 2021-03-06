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
           'BinaryFocalLoss',
           'FocalLoss',
           'ExtendedCrossEntropy',
           'DiceLoss',
           'SquaredDiceLoss',
           'MonsterLoss',
           'MonsterLossAux',
           'ExtendedMonsterLoss']

import numpy as np
import torch
from torch import sigmoid
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from torch.nn.functional import softmax


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
        loss = (torch.exp(-s) * F.l1_loss(yhat, y, reduction='none')) + s
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


class BinaryMaskLossSegmentation(MaskLossSegmentation):
    def forward(self, out, y):
        if not self.use_mask:
            loss = self.loss_fn(out, y)
        else:
            tgt, mask = torch.chunk(y, 2, dim=1)
            mask = mask.float()
            mask *= self.beta
            mask[mask == 0.] = 1.
            mask /= self.beta
            loss = torch.mean(mask * self.loss_fn(out, tgt, reduction='none'))
        return loss


class BinaryFocalLoss(BinaryMaskLossSegmentation):
    def __init__(self, beta=25., use_mask=False, weight=None, gamma=2.):
        super().__init__(beta, use_mask)
        self.weight = weight
        self.gamma = gamma

    def loss_fn(self, out, y, reduction='mean'):
        """ Taken from: https://github.com/catalyst-team/catalyst/, https://github.com/facebookresearch/fvcore """
        pred, _ = out
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")
        p_t = p * y + (1 - p) * (1 - y)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.weight is not None:
            weight_t = self.weight * y + (1 - self.weight) * (1 - y)
            loss = weight_t * loss
        if reduction == "mean":
            loss = loss.mean()
        if reduction == "sum":
            loss = loss.sum()
        if reduction == "batchwise_mean":
            loss = loss.sum(0)
        return loss


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
        """ https://github.com/alainjungo/reliability-challenges-uncertainty """
        logits, sigma = out
        dist = torch.distributions.Normal(logits, torch.exp(sigma))
        x_hat = dist.rsample((self.nsamp,))
        mc_prob = F.softmax(x_hat, dim=2).mean(dim=0)  # channel dim = 2 b/c samples
        return F.nll_loss(mc_prob.log(), y, weight=self.weight, reduction=reduction)

# The below is from Shuo Han's repo: https://gitlab.com/shan-deep-networks/pytorch-metrics/

def prob_encode(input):
    """Apply softmax or sigmoid.

    Args:
        input (torch.Tensor): Input tensor

    Returns:
        result (torch.Tensor): The result

    """
    result = softmax(input, dim=1) if input.shape[1] > 1 else sigmoid(input)
    return result


def one_hot(input, shape):
    """One hot encoding; torch does not have it as the current version

    Args:
        input (torch.LongTensor): The tensor to encode. The values should be
            "normalized" to 0 : num_labels

    Returns:
        result (torch.FloatTensor): The encoded tensor

    """
    result = torch.FloatTensor(shape).zero_()
    if input.is_cuda:
        result = result.cuda(device=input.device)
    result.scatter_(1, input, 1)
    return result


def _calc_dices(input, target, eps=0.001, keepdim=False):
    """Calculate dices for each sample and each channel

    Args:
        input (torch.FloatTensor): The input tensor
        target (torch.FloatTensor): The target tensor, one_hot encoded

    Returns:
        dices (torch.FloatTensor): The dices of each sample (first dim) and each
            channel (second dim)

    """
    spatial_dims = tuple(range(2 - len(input.shape), 0))
    intersection = torch.sum(input * target, dim=spatial_dims, keepdim=keepdim)
    sum1 = torch.sum(input, dim=spatial_dims, keepdim=keepdim)
    sum2 = torch.sum(target, dim=spatial_dims, keepdim=keepdim)
    dices = (2 * intersection + eps) / (sum1 + sum2 + eps)
    return dices


def _calc_squared_dices(input, target, eps=0.001):
    """Calculate squared dices for each sample and each channel

    Args:
        input (torch.FloatTensor): The input tensor
        target (torch.FloatTensor): The target tensor, one_hot encoded
        eps (float): The smoothing term preventing division by 0

    Returns:
        dices (torch.FloatTensor): The dices of each sample (first dim) and each
            channel (second dim)

    """
    spatial_dims = tuple(range(2 - len(input.shape), 0))
    intersection = torch.sum(input * target, dim=spatial_dims)
    sum1 = torch.sum(input ** 2, dim=spatial_dims)
    sum2 = torch.sum(target ** 2, dim=spatial_dims)
    dices = (2 * intersection + eps) / (sum1 + sum2 + eps)
    return dices


def calc_weighted_average(vals, weight):
    """Calculate weighted average along the second dim of values

    Args:
        vals (torch.Tensor): The values to weight; the first dim is samples
        weight (torch.Tensor): The 1d weights to apply to the second dim of vals

    Returns:
        result (torch.Tensor): The result

    """
    weight = weight[None, ...].repeat([vals.shape[0], 1])
    result = torch.mean(weight * vals)
    return result


def calc_dice_loss(input, target, weight=None, average=True, eps=0.001):
    """Calculate the dice loss

    Args:
        input (torch.Tensor): The input tensor
        target (torch.Tensor): The target tensor
        eps (float): The smoothing term preventing division by 0

    Return:
        dice (torch.Tensor): The weighted dice

    """
    dices = _calc_dices(input, target, eps=eps, keepdim=not average)
    if average:
        if weight is None:
            dice = torch.mean(dices)
        else:
            dice = calc_weighted_average(dices, weight)
    else:
        dice = dices
    return 1 - dice


def calc_squared_dice_loss(input, target, weight=None, eps=0.001):
    """Calculate the squared dice loss

    Args:
        input (torch.Tensor): The input tensor
        target (torch.Tensor): The target tensor
        eps (float): The smoothing term preventing division by 0

    Return:
        dice (torch.Tensor): The weighted dice

    """
    dices = _calc_squared_dices(input, target, eps=eps)
    if weight is None:
        dice = torch.mean(dices)
    else:
        dice = calc_weighted_average(dices, weight)
    return 1 - dice


def calc_dice(input, target, channel_indices=None, eps=0):
    """Calculate average Dice coefficients across samples and channels

    Args:
        input (torch.Tensor): The input tensor
        target (torch.Tensor): The target tensor
        channel_indices (list of int): The channels to calculate dices across.
            If None, use all channels
        eps (float): Small number preventing division by zero

    Returns:
        dice (torch.Tensor): The average Dice

    """
    input = prob_encode(input)
    if input.shape[1] > 2:
        input_seg = one_hot(torch.argmax(input, dim=1, keepdim=True), input.shape)
        target_onehot = one_hot(target, input.shape)
    else:
        input_seg = (input >= 0.5).float()
        target_onehot = target.float()
    if channel_indices is not None:
        input_seg = input_seg[:, channel_indices, ...]
        target_onehot = target_onehot[:, channel_indices, ...]
    dices = _calc_dices(input_seg, target_onehot, eps=eps)
    return torch.mean(dices)


class SquaredDiceLoss(_WeightedLoss):
    """ Wrapper of squared Dice loss. """
    def __init__(self, weight=None):
        super().__init__(weight=weight)

    def forward(self, input, target):
        input = prob_encode(input)
        target_onehot = one_hot(target, input.shape)
        return calc_squared_dice_loss(input, target_onehot, weight=self.weight)


class DiceLoss(_WeightedLoss):
    """ Wrapper of Dice loss. """
    def __init__(self, weight=None, average=True):
        super().__init__(weight=weight)
        self.average = average

    def forward(self, input, target):
        input = prob_encode(input)
        if input.shape[1] > 2:
            target_onehot = one_hot(target, input.shape)
        else:
            target_onehot = target.float()
        return calc_dice_loss(input, target_onehot, weight=self.weight,
                              average=self.average)


class MonsterLoss(BinaryMaskLossSegmentation):
    """ use focal, dice, and l1 loss together """
    def __init__(self, alpha=(1.,1.,1.), beta=25., use_mask=False, gamma=2., weight=None, use_l2=True):
        super().__init__(beta, use_mask)
        self.alpha = alpha
        self.weight = weight
        self.gamma = gamma
        self.use_l2 = use_l2

    def loss_fn(self, out, y, reduction='mean'):
        average = reduction == 'mean'
        pred, _ = out
        pred_t = torch.tanh(pred)

        if self.alpha[0] > 0.:
            if self.gamma == 0.:
                focal_loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=self.weight, reduction=reduction)
            else:
                ce_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")
                p = (pred_t + 1.) / 2.
                p_t = p * y + (1 - p) * (1 - y)
                focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
                if self.weight is not None:
                    weight_t = self.weight * y + (1 - self.weight) * (1 - y)
                    focal_loss = weight_t * focal_loss
                if average: focal_loss = focal_loss.mean()
        else:
            focal_loss = 0.

        if self.alpha[1] > 0.:
            p = (pred_t + 1.) / 2.
            dice_loss = calc_dice_loss(p, y, average=average)
        else:
            dice_loss = 0.

        if self.alpha[2] > 0.:
            y = (y * 2.) - 1.
            reg_loss = F.mse_loss(pred_t, y, reduction=reduction) if self.use_l2 else \
                       F.l1_loss(pred_t, y, reduction=reduction)
        else:
            reg_loss = 0.

        return self.alpha[0] * focal_loss + self.alpha[1] * dice_loss + self.alpha[2] * reg_loss


class ExtendedMonsterLoss(BinaryMaskLossSegmentation):
    def __init__(self, alpha=(1.,1.,1.), beta=25., use_mask=False, gamma=2., weight=None, n_samples=10,
                 use_l2=True, extended_regression=False):
        super().__init__(beta, use_mask)
        self.alpha = alpha
        self.weight = weight
        self.gamma = gamma
        self.nsamp = n_samples
        self.mlv = -13.816  # min log variance = ~log(1e-6)
        self.use_l2 = use_l2
        self.extended_regression = extended_regression

    def loss_fn(self, out, y, reduction='mean'):
        average = reduction == 'mean'
        pred, sigma = out
        sigma = torch.clamp_min(sigma, self.mlv)
        dist = torch.distributions.Normal(pred, torch.exp(sigma))
        mc_logs = dist.rsample((self.nsamp,))
        pred_t = torch.tanh(mc_logs.mean(dim=0))

        if self.alpha[0] > 0.:
            focal_loss = 0.
            if self.gamma == 0.:
                for mc_log in mc_logs:
                    focal_loss += F.binary_cross_entropy_with_logits(mc_log, y, pos_weight=self.weight, reduction=reduction)
            else:
                for mc_log in mc_logs:
                    ce_loss = F.binary_cross_entropy_with_logits(mc_log, y, reduction="none")
                    mc_pred_t = torch.tanh(mc_log)
                    p = (mc_pred_t + 1.) / 2.
                    p_t = p * y + (1. - p) * (1. - y)
                    focal_loss += (ce_loss * ((1. - p_t) ** self.gamma)) / self.nsamp
                    if self.weight is not None:
                        weight_t = self.weight * y + (1. - self.weight) * (1. - y)
                        focal_loss *= weight_t
            if average: focal_loss = focal_loss.mean()
        else:
            focal_loss = 0.

        if self.alpha[1] > 0.:
            p = (pred_t + 1.) / 2.
            dice_loss = calc_dice_loss(p, y, average=average)
        else:
            dice_loss = 0.

        if self.alpha[2] > 0.:
            y = (y * 2.) - 1.
            if self.extended_regression:
                if self.use_l2:
                    reg_loss = 0.5 * (torch.exp(-sigma) * F.mse_loss(pred_t, y, reduction='none') + sigma)
                else:
                    reg_loss = (torch.exp(-sigma) * F.l1_loss(pred_t, y, reduction='none')) + sigma
            else:
                reg_loss = F.mse_loss(pred_t, y, reduction='none') if self.use_l2 else F.l1_loss(pred_t, y, reduction='none')
            if average: reg_loss = torch.mean(reg_loss)
        else:
            reg_loss = 0.

        return self.alpha[0] * focal_loss + self.alpha[1] * dice_loss + self.alpha[2] * reg_loss


class MonsterLossAux(BinaryMaskLossSegmentation):
    """ use focal, dice, and l1 loss together """
    def __init__(self, alpha=(1.,1.,1.), beta=25., use_mask=False, gamma=2.,
                 weight=None, use_l2=True, balance_weights=(1.0, 0.4)):
        super().__init__(beta, use_mask)
        self.alpha = alpha
        self.weight = weight
        self.gamma = gamma
        self.use_l2 = use_l2
        self.balance_weights = balance_weights

    def loss_fn(self, out, y, reduction='mean'):
        average = reduction == 'mean'
        loss = 0.
        for bw, pred in zip(self.balance_weights, out):
            pred_t = torch.tanh(pred)

            if self.alpha[0] > 0.:
                if self.gamma == 0.:
                    focal_loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=self.weight, reduction=reduction)
                else:
                    ce_loss = F.binary_cross_entropy_with_logits(pred, y, reduction="none")
                    p = (pred_t + 1.) / 2.
                    p_t = p * y + (1 - p) * (1 - y)
                    focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
                    if self.weight is not None:
                        weight_t = self.weight * y + (1 - self.weight) * (1 - y)
                        focal_loss = weight_t * focal_loss
                    if average: focal_loss = focal_loss.mean()
            else:
                focal_loss = 0.

            if self.alpha[1] > 0.:
                p = (pred_t + 1.) / 2.
                dice_loss = calc_dice_loss(p, y, average=average)
            else:
                dice_loss = 0.

            if self.alpha[2] > 0.:
                y = (y * 2.) - 1.
                reg_loss = F.mse_loss(pred_t, y, reduction=reduction) if self.use_l2 else \
                           F.l1_loss(pred_t, y, reduction=reduction)
            else:
                reg_loss = 0.

            loss += bw * (self.alpha[0] * focal_loss + self.alpha[1] * dice_loss + self.alpha[2] * reg_loss)

        return loss
