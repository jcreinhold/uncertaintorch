#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uncertainunetv3

holds model definition for UncertainUnetv3

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: January 03, 2020
"""

__all__ = ['UncertainUnetv4']

from torch import nn
import torch.nn.functional as F

from ..learn import *
from .unet_tools import *
from .uncertainty_tools import *


class UncertainUnetv4(UncertainNet):
    def __init__(self, s=32, ic=1, oc=1, p=0.15, segnet=True, 
                 laplacian=True, aleatoric=True, concat=True, beta=25.):
        super().__init__(p, segnet, laplacian, aleatoric, concat, beta)
        c = 2 if concat else 1
        self.init = nn.Sequential(*conv(ic,s,3,1))
        self.start_0, self.start_1 = unet_list(s,s,s,(3,3,1),(3,3,1),1,1)
        self.start_2 = nn.Sequential(*conv(s,s*2,(3,3,1),(2,2,1)))
        self.down1_0, self.down1_1 = unet_list(s*2,s*2,s*2,(3,3,1),(3,3,1),1,1)
        self.down1_2 = nn.Sequential(*conv(s*2,s*4,(3,3,1),(2,2,1)))
        self.down2_0, self.down2_1 = unet_list(s*4,s*4,s*4,(3,3,1),(3,3,1),1,1)
        self.down2_2 = nn.Sequential(*conv(s*4,s*8,(3,3,1),(2,2,1)))
        self.bridge = nn.Sequential(*conv(s*8,s*4,(3,3,1)))
        self.up2_0, self.up2_1 = unet_up(s*4*c,s*4,s*2,(3,3,1),(3,3,1),1,1,(2,2,1),concat,True,(3,3,1))
        self.up1_0, self.up1_1 = unet_up(s*2*c,s*2,s,(3,3,1),(3,3,1),1,1,(2,2,1),concat,True,(3,3,1))
        self.end_0, self.end_1 = unet_up(s*c,s,s,(3,3,1),(3,3,1),1,1,(2,2,1),concat,True,(3,3,1))
        self.syn = nn.Sequential(*conv(s+ic,s,3,1), nn.Conv3d(s,oc,1))
        self.unc = nn.Sequential(*conv(s+ic,s,3,1), nn.Conv3d(s,oc,1))

    def fwd_bayesian_segnet(self, x):
        orig = x.clone()
        x = self.init(x)
        x = self.start_0(x)
        x = self.start_1(x)
        d1 = x.clone()
        x = self.dropout(x)  # not in bayesian segnet
        x = self.start_2(x)
        x = self.down1_0(x)
        x = self.down1_1(x)
        d2 = x.clone()
        x = self.dropout(x)
        x = self.down1_2(x)
        x = self.down2_0(x)
        x = self.down2_1(x)
        d3 = x.clone()
        x = self.dropout(x)
        x = self.down2_2(x)
        x = self.bridge(x)
        x = self.up2_0(x)
        x = self.catadd(x, d3)
        x = self.up2_1(x)
        x = self.dropout(x)
        x = self.up1_0(x)
        x = self.catadd(x, d2)
        x = self.up1_1(x)
        x = self.dropout(x)
        x = self.end_0(x)
        x = self.catadd(x, d1)
        x = self.end_1(x)
        x = self.dropout(x)  # not in bayesian segnet
        x = self.cat(x,orig)
        yhat = self.syn(x)
        s = self.unc(x)
        return yhat, s

    def fwd_full_bayesian(self, x):
        orig = x.clone()
        x = self.init(x)
        x = self.dropout(x)
        x = self.start_0(x)
        x = self.dropout(x)
        x = self.start_1(x)
        d1 = x.clone()
        x = self.dropout(x)
        x = self.start_2(x)
        x = self.dropout(x)
        x = self.down1_0(x)
        x = self.dropout(x)
        x = self.down1_1(x)
        d2 = x.clone()
        x = self.dropout(x)
        x = self.down1_2(x)
        x = self.dropout(x)
        x = self.down2_0(x)
        x = self.dropout(x)
        x = self.down2_1(x)
        d3 = x.clone()
        x = self.dropout(x)
        x = self.down2_2(x)
        x = self.dropout(x)
        x = self.bridge(x)
        x = self.dropout(x)
        x = self.up2_0(x)
        x = self.dropout(x)
        x = self.catadd(x, d3)
        x = self.up2_1(x)
        x = self.dropout(x)
        x = self.up1_0(x)
        x = self.dropout(x)
        x = self.catadd(x, d2)
        x = self.up1_1(x)
        x = self.dropout(x)
        x = self.end_0(x)
        x = self.dropout(x)
        x = self.catadd(x, d1)
        x = self.end_1(x)
        x = self.dropout(x)
        x = self.cat(x,orig)
        yhat = self.syn(x)
        s = self.unc(x)
        return yhat, s
