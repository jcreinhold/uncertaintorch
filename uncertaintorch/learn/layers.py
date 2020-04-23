#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
layers

holds layers definitions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = [
    'ChannelAttention',
    'GeLU',
    'SelfAttention',
    'Upconv3d'
]

import torch
from torch import nn
import torch.nn.functional as F


def pixel_shuffle_3d(x, scale_factor):
    if isinstance(scale_factor, int): scale_factor = [scale_factor] * 3
    batch_size, channels, in_depth, in_height, in_width = x.size()
    channels //= (scale_factor[0] * scale_factor[1] * scale_factor[2])
    out_depth = in_depth * scale_factor[0]
    out_height = in_height * scale_factor[1]
    out_width = in_width * scale_factor[2]
    input_view = x.contiguous().view(
        batch_size, channels, scale_factor[0], scale_factor[1], scale_factor[2], in_depth, in_height, in_width)
    shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class Upconv3d(nn.Module):
    def __init__(self, ic, oc, scale_factor=2, full=False, k=1):
        super().__init__()
        if isinstance(scale_factor, int): scale_factor = [scale_factor] * 3
        self.sf = scale_factor
        sf = (scale_factor[0] * scale_factor[1] * scale_factor[2])
        pad = k//2 if isinstance(k,int) else tuple([ks//2 for p in zip(reversed(k),reversed(k)) for ks in p])
        if isinstance(k,int): self.pad = None if  k < 3 else nn.ReplicationPad3d(pad)
        if isinstance(k,tuple): self.pad = None if all([p == 0 for p in pad]) else nn.ReplicationPad3d(pad)
        self.conv = nn.Conv3d(ic, oc*sf, k, bias=False)
        self.full = full
        if full:
            self.bn = nn.BatchNorm3d(oc)
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.pad is not None: x = self.pad(x)
        x = pixel_shuffle_3d(self.conv(x), self.sf)
        if self.full: out = self.act(self.bn(x))
        return x


class GeLU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SelfAttention(nn.Module):
    """ Self attention layer for 2d (implementation inspired by fastai library) """
    def __init__(self, n_channels:int):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels//8, 1))
        self.key   = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels//8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels, 1))
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=8):
        super().__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.mlp = nn.Sequential(
            nn.Linear(2 * gate_channels, gate_channels // reduction_ratio, bias=False),
            nn.BatchNorm1d(gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels))

    def forward(self, x:torch.Tensor):
        xf = x.flatten(start_dim=2)
        mp, _ = xf.max(dim=2)
        ap    = xf.mean(dim=2)
        y = self.mlp(torch.cat((mp, ap), dim=1))
        scale = torch.sigmoid(y)
        for _ in range((x.ndimension()-2)): scale = scale.unsqueeze(-1)
        return x * scale.expand_as(x)
