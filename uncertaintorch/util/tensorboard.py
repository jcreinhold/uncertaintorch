#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tensorboard

holds tensorboard helper functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['TrainTB', 'ValidTB']

import torch


def normalize(x):
    dim = x.dim()
    xmin, xmax = x.clone(), x.clone()
    for i in range(1,dim):
        xmin, _ = xmin.min(dim=i, keepdim=True)
        xmax, _ = xmax.max(dim=i, keepdim=True)
    return (x - xmin) / (xmax - xmin)


def norm(x,d=-1):
    return torch.norm(x, p=2, dim=d, keepdim=True)


class TB:
    def __init__(self, writer):
        self.W = writer

    def histogram_weights(self, model, epoch, model_name):
        """ write histogram of weights to tensorboard """
        for (name, values) in model.named_parameters():
            self.W.add_histogram(tag=f'{model_name}/{name}',
                                 values=values.clone().detach().cpu(),
                                 global_step=epoch)

class TrainTB(TB):
    def __call__(self, loss, i, t, n_batches, tb_rate, unc=None, seg=None):
        loss_rate, img_rate, hist_rate = tb_rate
        if i % loss_rate == 0:
            j = ((t-1) * n_batches) + i
            loss = loss.item() if hasattr(loss, 'item') else loss
            self.W.add_scalar('loss/train', loss, j)
            if seg is not None:
                ds, isbi15 = seg
                ds = ds.item() if hasattr(ds, 'item') else ds
                isbi15 = isbi15.item() if hasattr(isbi15, 'item') else isbi15
                self.W.add_scalar('dice/train', ds, j)
                self.W.add_scalar('isbi15/train', isbi15, j)
            if unc is not None:
                if len(unc) == 3:
                    ep, al, sb = unc
                    en, ep2 = None, None
                else:
                    ep, en, al, ep2, sb = unc
                self.W.add_scalar('epistemic/train', ep, j)
                self.W.add_scalar('aleatoric/train', al, j)
                self.W.add_scalar('scibilic/train',  sb, j)
                if en is not None:
                    self.W.add_scalar('entropy/train', en, j)
                    self.W.add_scalar('epistemic2/train', ep2, j)


class ValidTB(TB):
    def __call__(self, loss, i, t, n_batches, tb_rate, x, y, pred, model, unc=None, seg=None, nv=1):
        loss_rate, img_rate, hist_rate = tb_rate
        if i % loss_rate == 0:
            j = ((t-1) * n_batches) + i
            loss = loss.item() if hasattr(loss, 'item') else loss
            self.W.add_scalar('loss/valid', loss, j)
            if seg is not None:
                ds, isbi15 = seg
                ds = ds.item() if hasattr(ds, 'item') else ds
                isbi15 = isbi15.item() if hasattr(isbi15, 'item') else isbi15
                self.W.add_scalar('dice/valid', ds, j)
                self.W.add_scalar('isbi15/valid', isbi15, j)
            if unc is not None:
                if len(unc) == 6:
                    epim, alim, sbim, ep, al, sb = unc
                    enim, en = None, None
                    ep2im, ep2 = None, None
                else:
                    epim, enim, alim, ep2im, sbim, ep, en, al, ep2, sb = unc
                self.W.add_scalar('epistemic/valid', ep, j)
                self.W.add_scalar('aleatoric/valid', al, j)
                self.W.add_scalar('scibilic/valid',  sb, j)
                if en is not None and ep2 is not None:
                    self.W.add_scalar('entropy/valid', en, j)
                    self.W.add_scalar('epistemic2/valid', ep2, j)
        plot_img = i == 0 and (t % img_rate) == 0
        if plot_img:
            if y.ndim == 5:
                sn = (x.size(4) // 2)  # slice number
                source = normalize(x[:nv,0:1,:,:,sn])
                target = normalize(y[:nv,0:1,:,:,sn]) if seg is None else \
                         y[:nv,:,:,sn].unsqueeze(1) if y.ndim == 4 else y[:nv,0:1,:,:,sn]
                prediction = normalize(pred[:nv,0:1,:,:,sn]) if seg is None else \
                             pred[:nv,:,:,sn].unsqueeze(1) if pred.ndim == 4 else pred[:nv,0:1,:,:,sn]
                if unc is not None:
                    epim = epim[:nv,:,:,:,sn]
                    alim = alim[:nv,:,:,:,sn]
                    sbim = sbim[:nv,:,:,:,sn]
                    if enim is not None and ep2im is not None:
                        enim = enim[:nv,:,:,:,sn]
                        ep2im = ep2im[:nv,:,:,:,sn]
            else:
                source = normalize(x[:nv,0:1,...])
                target = normalize(y[:nv,0:1,...]) if seg is None else y[:nv,0:1,...]
                prediction = normalize(pred[:nv,0:1,...]) if seg is None else pred[:nv,0:1,...]
                if unc is not None:
                    epim = epim[:nv,...]
                    alim = alim[:nv,...]
                    sbim = sbim[:nv,...]
                    if enim is not None:
                        enim = enim[:nv,...]
                        ep2im = ep2im[:nv, ...]
            self.W.add_images('input/source', source, t, dataformats='NCHW')
            self.W.add_images('input/target', target, t, dataformats='NCHW')
            self.W.add_images('output/prediction', prediction, t, dataformats='NCHW')
            if unc is not None:
                self.W.add_images('output/epistemic', epim, t, dataformats='NCHW')
                self.W.add_images('output/aleatoric', alim, t, dataformats='NCHW')
                self.W.add_images('output/scibilic',  sbim, t, dataformats='NCHW')
                if enim is not None:
                    self.W.add_images('output/entropy', enim, t, dataformats='NCHW')
                    self.W.add_images('output/epistemic2', ep2im, t, dataformats='NCHW')
        plot_hist = i == 0 and (t % hist_rate) == 0
        if plot_hist:
            self.histogram_weights(model, t, 'weights')
