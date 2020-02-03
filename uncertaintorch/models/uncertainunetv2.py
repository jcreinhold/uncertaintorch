#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uncertainunetv2

holds model definition for UncertainUnetv2

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['UncertainUnetv2']

from torch import nn

from .unet_tools import *
from .uncertainty_tools import *


class UncertainUnetv2(UncertainNet):
    def __init__(self, s=32, ic=1, oc=1, p=0.15, segnet=True, 
                 laplacian=True, aleatoric=True, concat=True, beta=25.):
        super().__init__(p, segnet, laplacian, aleatoric, concat)
        c = 2 if concat else 1
        self.start_0, self.start_1 = unet_list(ic,s,s,3,3,1,2)
        self.down1_0, self.down1_1 = unet_list(s,s*2,s*2,3,3,1,2)
        self.down2_0, self.down2_1 = unet_list(s*2,s*4,s*4,3,3,1,2)
        self.bridge = nn.Sequential(*conv(s*4,s*4))
        self.up2_0, self.up2_1 = unet_up(s*4*c,s*4,s*2,3,3,1,1,2,concat)
        self.up1_0, self.up1_1 = unet_up(s*2*c,s*2,s,3,3,1,1,2,concat)
        self.end_0, self.end_1 = unet_up(s*c,s,s,3,3,1,1,2,concat)
        self.syn = nn.Sequential(*conv(s+ic,s,3,1), nn.Conv3d(s,oc,1))
        self.unc = nn.Sequential(*conv(s+ic,s,3,1), nn.Conv3d(s,oc,1))

    def fwd_bayesian_segnet(self,x):
        orig = x.clone()
        x = self.start_0(x)
        d1 = x.clone()
        x = self.start_1(x)
        x = self.down1_0(x)
        d2 = x.clone()
        x = self.dropout(x)
        x = self.down1_1(x)
        x = self.down2_0(x)
        d3 = x.clone()
        x = self.dropout(x)
        x = self.down2_1(x)
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
        x = self.cat(x, orig)
        yhat = self.syn(x)
        s = self.unc(x)
        return yhat, s

    def fwd_full_bayesian(self,x):
        raise NotImplementedError

