#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cbr

just a regular ol' series of conv-bn-relu
(with one skip connection)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: March 10, 2020
"""

__all__ = ['CBR']

import torch
from torch import nn
import torch.nn.functional as F

from .unet_tools import conv
from ..learn import *


class CBR(nn.Module):
    def __init__(self, s=32, ic=1, oc=1, p=0., beta=25., bayesian=False, laplacian=False):
        super().__init__()
        self.p = p
        self.conv1 = nn.Sequential(*conv(ic, s, 7))
        self.conv2 = nn.Sequential(*conv(s, s*2, 5))
        self.conv3 = nn.Sequential(*conv(s*2, s*4, 3))
        self.conv4 = nn.Sequential(*conv(s*4, s*4, 3))
        self.syn = nn.Sequential(*conv(s*4+s, s, 3),
                                 nn.Conv3d(s, oc,1))
        self.unc = nn.Sequential(*conv(s*4+s, s, 3),
                                 nn.Conv3d(s, oc,1))
        if bayesian:
            self.criterion = LaplacianDiagLoss(beta) if laplacian else GaussianDiagLoss(beta)
        else:
            self.criterion = L1OnlyLoss(beta) if laplacian else MSEOnlyLoss(beta)

    def dropout(self, x):
        use_dropout = self.bayesian or self.training
        return F.dropout3d(x, self.p, training=use_dropout, inplace=False)

    @staticmethod
    def cat(x, y): return torch.cat((x, y), dim=1)

    def forward(self, x):
        x = self.conv1(x)
        start = x.clone()
        x = self.dropout(x)
        x = self.dropout(self.conv2(x))
        x = self.dropout(self.conv3(x))
        x = self.dropout(self.conv4(x))
        x = self.cat(x, start)
        yhat = self.syn(x)
        s = self.unc(x)
        return yhat, s
