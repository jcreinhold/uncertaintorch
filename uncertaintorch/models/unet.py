#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
unet

just a regular ol' 3d Unet

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['Unet']

import torch
from torch import nn
import torch.nn.functional as F

from .unet_tools import *


class Unet(nn.Module):
    def __init__(self, s=32, ic=1, oc=1, laplacian=True):
        super().__init__()
        self.start = unet_block(ic,s,s,3,3)
        self.down1 = unet_block(s,s*2,s*2,3,3)
        self.down2 = unet_block(s*2,s*4,s*4,3,3)
        self.bridge = unet_block(s*4,s*8,s*4,3,3)
        self.up2 = unet_block(s*8,s*4,s*2,5,3)
        self.up1 = unet_block(s*4,s*2,s,5,3)
        self.final = nn.Sequential(*conv(s*2,s,5),
                                   *conv(s,s,3),
                                   nn.Conv3d(s,oc,1))
        self.criterion = nn.L1Loss() if laplacian else nn.MSELoss()

    def forward(self,x):
        r = [self.start(x)]
        r.append(self.down1(F.max_pool3d(r[-1],2)))
        r.append(self.down2(F.max_pool3d(r[-1],2)))
        x = F.interpolate(self.bridge(F.max_pool3d(r[-1],2)),size=r[-1].shape[2:])
        x = F.interpolate(self.up2(torch.cat((x,r[-1]),dim=1)),size=r[-2].shape[2:])
        x = F.interpolate(self.up1(torch.cat((x,r[-2]),dim=1)),size=r[-3].shape[2:])
        x = self.final(torch.cat((x,r[-3]),dim=1))
        return x



