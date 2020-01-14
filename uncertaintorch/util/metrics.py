#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics

holds metrics to evaluate results

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: January 14, 2019
"""

__all__ = ['jaccard', 'dice', 'largest_cc']

from torch import Tensor


def jaccard(x:Tensor, y:Tensor):
    xm, ym = (x > 0), (y > 0)
    intersection = (xm & ym).sum().float()
    union = (xm | ym).sum().float()
    if union == 0.: return 1.
    return intersection / union


def dice(x:Tensor, y:Tensor):
    xm, ym = (x > 0), (y > 0)
    intersection = (xm & ym).sum().float()
    cardinality = (xm.sum() + ym.sum()).float()
    if cardinality == 0.: return 1.
    return 2 * intersection / cardinality


def largest_cc(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    lcc = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    return lcc

