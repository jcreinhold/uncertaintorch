#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
imshow

wrapper for imshow for medical images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['imshow']

import numpy as np


def imshow(x, ax, title, n_rot=3):
    ax.imshow(np.rot90(x,n_rot), aspect='equal', cmap='gray')
    ax.set_title(title,fontsize=22)
    ax.axis('off');

