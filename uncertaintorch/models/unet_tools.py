#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
unet_tools

miscellaneous functions for unet definitions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['conv','unet_block','unet_list','unet_up']

from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..learn import *
from ..util import *

activation = partial(nn.ReLU, inplace=False)


def conv(i,o,k=3,s=1):
    pad = k//2 if isinstance(k,int) else tuple([ks//2 for p in zip(reversed(k),reversed(k)) for ks in p])
    if isinstance(k,int): c = [] if  k < 3 else [nn.ReplicationPad3d(pad)]
    if isinstance(k,tuple): c = [] if all([p == 0 for p in pad]) else [nn.ReplicationPad3d(pad)]
    c.extend([nn.Conv3d(i,o,k,s,bias=False), nn.BatchNorm3d(o), activation()])
    return c


def unet_block(i,m,o,k1,k2):
    return nn.Sequential(*conv(i,m,k1),*conv(m,o,k2))


def unet_list(i,m,o,k1,k2,s1=1,s2=1):
    layers = [nn.Sequential(*conv(i,m,k1,s1)),
              nn.Sequential(*conv(m,o,k2,s2))]
    return nn.ModuleList(layers)


def unet_up(i,m,o,k1,k2,s1=1,s2=1,scale_factor=2,cat=True,full=False,upk=1):
    c = 2 if cat else 1
    layers = [Upconv3d(i//c,i//c,scale_factor,full,upk),
              nn.Sequential(*conv(i,m,k1,s1), *conv(m,o,k2,s2))]
    return nn.ModuleList(layers)
