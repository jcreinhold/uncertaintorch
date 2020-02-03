#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
util

holds general utility functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['init_fn',
           'num_params',
           'to_np',
           'to_cpu',
           'list_to_np',
           'list_to_item']

import random

import numpy as np
import torch


def init_fn(worker_id):
    random.seed((torch.initial_seed() + worker_id) % (2**32))
    np.random.seed((torch.initial_seed() + worker_id) % (2**32))


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_cpu(x):
    return x.cpu().detach().squeeze() if isinstance(x, torch.Tensor) else x


def to_np(x):
    return to_cpu(x).numpy() if isinstance(x, torch.Tensor) else x


def list_to_np(lst):
    return list(map(lambda x: to_np(x), lst))


def list_to_item(lst):
    return list(map(lambda x: x.item(), lst))

