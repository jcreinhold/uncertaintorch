#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models

holds model definitions for SPIE 2020 experiments

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['tidy_losses', 'tidy_uncertainty']

import pandas as pd


def tidy_losses(train, valid):
    out = {'epoch': [], 'type': [], 'value': [], 'phase': []}
    for i, (tl,vl) in enumerate(zip(train,valid),1):
        for tli in tl:
            out['epoch'].append(i)
            out['type'].append('loss')
            out['value'].append(tli)
            out['phase'].append('train')
        for vli in vl:
            out['epoch'].append(i)
            out['type'].append('loss')
            out['value'].append(vli)
            out['phase'].append('valid')
    return pd.DataFrame(out)

def tidy_uncertainty(ep, al, sb):
    out = {'epoch': [], 'type': [], 'value': [], 'phase': []}
    for i, (epi, ali, sbi) in enumerate(zip(ep, al, sb)):
        phase = 'train' if i == 0 else 'valid'
        for j, (epij,alij,sbij) in enumerate(zip(epi,ali,sbi),1):
            for epijk in epij:
                out['epoch'].append(j)
                out['type'].append('epistemic')
                out['value'].append(epijk)
                out['phase'].append(phase)
            for alijk in alij:
                out['epoch'].append(j)
                out['type'].append('aleatoric')
                out['value'].append(alijk)
                out['phase'].append(phase)
            for sbijk in sbij:
                out['epoch'].append(j)
                out['type'].append('scibilic')
                out['value'].append(sbijk)
                out['phase'].append(phase)
    return pd.DataFrame(out)
