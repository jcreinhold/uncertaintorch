#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
deeplabv3

3d deeplabv3-derived model for synthesis or segmentation

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['DeepLab3d']

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..learn import *
from .unet_tools import *
from .resnet import *
from .resnet import BASE_WIDTH, EXPANSION

RESNET= {'resnet18': resnet18, 'resnet34': resnet34,'resnet50': resnet50,'resnet101': resnet101}


class DeepLab3d(nn.Module):
    def __init__(self, backbone_name='resnet101', ic=1, oc=1, p=0., beta=25.,
                 replace_stride_with_dilation=(False, True, True),
                 bayesian=False, laplacian=True, segmentation=None):
        super().__init__()
        nrswd = 4 - sum([int(rswd) for rswd in replace_stride_with_dilation])
        self.width = width = (BASE_WIDTH * nrswd) * EXPANSION
        self.backbone = RESNET[backbone_name](in_channels=ic,
            replace_stride_with_dilation=replace_stride_with_dilation)
        self.head = DeepLabHead(width, BASE_WIDTH, mid_channels=width)
        self.orig_conv = nn.Conv3d(ic, 1, 1)
        self.mid_conv = nn.Conv3d(width//2, width//4, 1)  #TODO: figure out why //2 on input
        self.end_1 = unet_block(BASE_WIDTH+width//4,BASE_WIDTH,BASE_WIDTH,3,3)
        self.end_2 = unet_block(BASE_WIDTH+1,BASE_WIDTH,BASE_WIDTH,3,3)
        self.syn = nn.Sequential(*conv(BASE_WIDTH,BASE_WIDTH,3,1), nn.Conv3d(BASE_WIDTH,oc,1))
        self.unc = nn.Sequential(*conv(BASE_WIDTH,BASE_WIDTH,3,1), nn.Conv3d(BASE_WIDTH,oc,1))
        self.p = p
        self.bayesian = bayesian
        self.segmentation = segmentation
        if bayesian:
            if segmentation:
                self.criterion = ExtendedCrossEntropy(beta, **segmentation)
            else:
                self.criterion = LaplacianDiagLoss(beta) if laplacian else GaussianDiagLoss(beta)
        else:
            if segmentation is not None:
                self.criterion = FocalLoss(beta, **segmentation)
            else:
                self.criterion = L1OnlyLoss(beta) if laplacian else MSEOnlyLoss(beta)

    def dropout(self,x):
        use_dropout = self.bayesian or self.training
        return F.dropout3d(x, self.p, training=use_dropout, inplace=False)

    @staticmethod
    def cat(x,r): return torch.cat((x, r), dim=1)

    @staticmethod
    def interp(x,r): return F.interpolate(x, size=r.shape[2:], mode='trilinear', align_corners=True)

    def interpcat(self,x,r): return self.cat(self.interp(x, r), r)

    def forward(self, x):
        orig = self.orig_conv(x)
        x, mid = self.backbone(x)  # dropout already in backbone fwd pass
        mid = self.mid_conv(mid)
        x = self.head(x)
        x = self.dropout(x)
        x = self.interpcat(x, mid)
        x = self.end_1(x)
        x = self.dropout(x)
        x = self.interpcat(x,orig)
        x = self.end_2(x)
        yhat, s = self.syn(x), self.unc(x)
        return yhat, s


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, mid_channels=128):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [6, 12, 18], out_channels=mid_channels),
            nn.ReplicationPad3d(1),
            nn.Conv3d(mid_channels, mid_channels, 3, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.ReplicationPad3d(dilation),
            nn.Conv3d(in_channels, out_channels, 3, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=128):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

