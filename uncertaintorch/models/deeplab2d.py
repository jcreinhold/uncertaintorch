#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
deeplab2d

2d deeplabv3-derived model for synthesis or segmentation

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: January 14, 2019
"""

__all__ = ['DeepLab2d']

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..learn import *
from .unet_tools import *


def conv(i,o,k):
    return (nn.Conv2d(i,o,k,padding=k//2,bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True))


def bottleneck(i,b,o,k):
    return (nn.Conv2d(i, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.Conv2d(b, b, k, padding=k//2, bias=False),
            nn.BatchNorm2d(b),
            nn.Conv2d(b, o, 1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True))


class DeepLab2d(nn.Module):
    def __init__(self, ic, oc, pretrained=True, use_large=True, nc=64, p=0.05, bayesian=False):
        super().__init__()
        self.p = p
        self.bayesian = bayesian
        backbone = torchvision.models.segmentation.deeplabv3_resnet101 if use_large else \
                   torchvision.models.segmentation.deeplabv3_resnet51
        self.start = nn.ModuleList([
            nn.Sequential(*conv(ic, nc, 3)),
            nn.Sequential(*bottleneck(nc, nc//2, nc, 3))])
        self.stem = nn.Sequential(*bottleneck(nc, nc//2, 3, 3))
        self.bb = backbone(pretrained=pretrained)
        self.bb.classifier[0].project[3] = nn.Dropout2d(0.2)  # change dropout to channel dropout
        del self.bb.classifier[4]
        self.bb.aux_classifier = None
        self.end = nn.Sequential(*bottleneck(256+ic, nc//2, nc, 3))
        self.last = nn.Sequential(*bottleneck(nc+ic, nc//2, nc, 3), nn.Conv2d(nc, oc, 1))
        if bayesian:
            if segmentation:
                self.criterion = ExtendedCrossEntropy(beta)
            else:
                self.criterion = LaplacianDiagLoss(beta) if laplacian else GaussianDiagLoss(beta)
        else:
            if segmentation:
                self.criterion = FocalLoss(beta)
            else:
                self.criterion = L1OnlyLoss(beta) if laplacian else MSEOnlyLoss(beta)

    def dropout(self, x):
        use_dropout = self.training if not self.bayesian else True
        return F.dropout2d(x, p=self.p, training=use_dropout)

    @staticmethod
    def cat(x,r): return torch.cat((x, r), dim=1)

    def forward(self, x):
        orig = x.clone()
        for s in self.start:
            x = self.dropout(s(x))
        x = self.stem(x)
        x = self.dropout(self.bb(x)['out'])  # torchvision net returns dict
        x = self.dropout(self.end(torch.cat((x,orig),dim=1)))
        x = self.last(torch.cat((x,orig),dim=1))
        return x

    def freeze_backbone(self):
        for param in self.bb.backbone.parameters():
            param.requires_grad = False

    def freeze_classifier(self):
        for param in self.bb.classifier.parameters():
            param.requires_grad = False

