#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uncertainunetv1

holds model definition for UncertainUnet

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['UncertainUnetv1']

from torch import nn
import torch.nn.functional as F

from .uncertainty_tools import *
from .unet_tools import *


class UncertainUnetv1(UncertainNet):
    def __init__(self, s=32, ic=1, oc=1, p=0.15, segnet=True,
                 laplacian=True, aleatoric=True, concat=True, beta=25.):
        super().__init__(p, segnet, laplacian, aleatoric, concat, beta)
        self.start_0, self.start_1 = unet_list(ic,s,s,3,3)
        self.down1_0, self.down1_1 = unet_list(s,s*2,s*2,3,3)
        self.down2_0, self.down2_1 = unet_list(s*2,s*4,s*4,3,3)
        self.bridge_0, self.bridge_1 = unet_list(s*4,s*8,s*4,3,3)
        self.up2_0, self.up2_1 = unet_list(s*8,s*4,s*2,5,3)
        self.up1_0, self.up1_1 = unet_list(s*4,s*2,s,5,3)
        self.end_0, self.end_1 = unet_list(s*2,s,s,5,3)
        self.syn = nn.Sequential(*conv(s+ic,s,3,1), nn.Conv3d(s,oc,1))
        self.unc = nn.Sequential(*conv(s+ic,s,3,1), nn.Conv3d(s,oc,1))

    def fwd_bayesian_segnet(self, x):
        orig = x.clone()
        x = self.start_0(x)
        x = self.start_1(x)
        d1 = x.clone()
        x = F.max_pool3d(x,2)
        x = self.down1_0(x)
        x = self.down1_1(x)
        d2 = x.clone()
        x = self.dropout(x)
        x = F.max_pool3d(x,2)
        x = self.down2_0(x)
        x = self.down2_1(x)
        d3 = x.clone()
        x = self.dropout(x)
        x = F.max_pool3d(x,2)
        x = self.bridge_0(x)
        x = self.bridge_1(x)
        x = self.dropout(x)
        x = self.interpcat(x, d3)
        x = self.up2_0(x)
        x = self.up2_1(x)
        x = self.dropout(x)
        x = self.interpcat(x, d2)
        x = self.up1_0(x)
        x = self.up1_1(x)
        x = self.dropout(x)
        x = self.interpcat(x, d1)
        x = self.end_0(x)
        x = self.end_1(x)
        x = self.cat(x, orig)
        yhat = self.syn(x)
        s = F.softplus(self.unc(x))
        return yhat, s

    def fwd_full_bayesian(self,x):
        orig = x.clone()
        x = self.dropout(self.start_0(x))
        x = self.dropout(self.start_1(x))
        d1 = x.clone()
        x = F.max_pool3d(x,2)
        x = self.dropout(self.down1_0(x))
        x = self.dropout(self.down1_1(x))
        d2 = x.clone()
        x = F.max_pool3d(x,2)
        x = self.dropout(self.down2_0(x))
        x = self.dropout(self.down2_1(x))
        d3 = x.clone()
        x = F.max_pool3d(x,2)
        x = self.dropout(self.bridge_0(x))
        x = self.dropout(self.bridge_1(x))
        x = self.interpcat(x, d3)
        x = self.dropout(self.up2_0(x))
        x = self.dropout(self.up2_1(x))
        x = self.interpcat(x, d2)
        x = self.dropout(self.up1_0(x))
        x = self.dropout(self.up1_1(x))
        x = self.interpcat(x, d1)
        x = self.dropout(self.end_0(x))
        x = self.dropout(self.end_1(x))
        x = self.cat(x, orig)
        yhat = self.syn(x)
        s = self.unc(x)
        return yhat, s

