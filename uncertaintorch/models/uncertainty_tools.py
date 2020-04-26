#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models

holds model definitions for SPIE 2020 experiments

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['UncertainBinarySegNet',
           'UncertainNet',
           'turn_on_dropout_layers']

import torch
from torch import nn
import torch.nn.functional as F

from ..learn import *
from ..util import *

class UncertainBinarySegNet:

    def binary_segmentation_uncertainty_predict(self, x, n_samp=50, eps=1e-6):
        logits, sigmas = [], []
        for _ in range(n_samp):
            logit, sigma = self.forward(x)
            logits.append(logit.detach().cpu())
            sigmas.append(sigma.detach().cpu())
        logits = torch.stack(logits)
        logit = logits.mean(dim=0)
        probits = torch.sigmoid(logits)
        epistemic = probits.var(dim=0, unbiased=True)
        probit = probits.mean(dim=0)
        entropy = -1 * (probit * (probit + eps).log() + ((1 - probit) * (1 - probit + eps).log()))  # entropy
        sigmas = torch.clamp_min(torch.stack(sigmas), -13.816)  # ~log(1e-6)
        sigma = sigmas.mean(dim=0)
        sigmas = torch.exp(sigmas)
        aleatoric = sigmas.mean(dim=0)
        epistemic2 = sigmas.var(dim=0, unbiased=True)
        return (logit, sigma, epistemic, entropy, aleatoric, epistemic2)

    def get_binary_segmentation_metrics(self, x, y, n_samp=50, eps=1e-6):
        """ get segmentation uncertainties and other metrics during training for analysis """
        state = self.training
        self.eval()
        with torch.no_grad():
            y = y.detach().cpu()
            logit, sigma, ep, en, al, ep2 = self.binary_segmentation_uncertainty_predict(x, n_samp, eps)
            if self.criterion.weight is not None:
                device = self.criterion.weight.device
                self.criterion.weight = self.criterion.weight.cpu()
            loss = self.criterion((logit, sigma), y)
            sb = ep / (al + eps)
            eu, au = ep.mean(), al.mean()
            nu = en.mean()
            su = sb.mean()
            eu2 = ep2.mean()
            pred = (logit >= 0)
            ds, js = list_to_np((dice(pred, y), jaccard(pred, y)))
        self.train(state)
        if self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(device)
        return loss, pred, (ep, en, al, ep2, sb), (eu, nu, au, eu2, su), (ds, js)


class UncertainNet(nn.Module):
    def __init__(self, p, segnet, laplacian, bayesian=True, concat=True, beta=25.):
        super().__init__()
        self.p = p
        self.segnet = segnet
        self.laplacian = laplacian
        self.bayesian = bayesian
        self.concat = concat
        if bayesian:
            self.criterion = LaplacianDiagLoss(beta) if laplacian else GaussianDiagLoss(beta)
        else:
            self.criterion = L1OnlyLoss(beta) if laplacian else MSEOnlyLoss(beta)
        self._fwd = self.fwd_bayesian_segnet if self.segnet else self.fwd_full_bayesian

    def forward(self,x):
        return self._fwd(x)

    def dropout(self,x):
        use_dropout = self.bayesian or self.training
        return F.dropout3d(x, self.p, training=use_dropout, inplace=False)

    @staticmethod
    def interp(x,r): return F.interpolate(x, size=r.shape[2:])

    @staticmethod
    def lininterp(x,r): return F.interpolate(x, size=r.shape[2:], mode='trilinear', align_corners=True)

    @staticmethod
    def cat(x,r): return torch.cat((x, r), dim=1)

    def interpcat(self,x,r): return self.cat(self.interp(x, r), r)

    def catadd(self,x,r):
        if x.shape[2:] != r.shape[2:]:
            x = self.lininterp(x, r)
        if self.concat:
            x = self.cat(x,r)
        else:
            x += r
        return x

    def calc_uncertainty(self, yhat, s):
        epistemic = yhat.var(dim=0, unbiased=True)
        aleatoric = torch.mean(torch.exp(s),dim=0) if not self.laplacian else \
                    torch.mean(2*torch.exp(s)**2,dim=0)
        return epistemic, aleatoric

    def predict(self, x, n_samp=50):
        out = [self.forward(x) for _ in range(n_samp)]
        yhat = torch.stack([o[0] for o in out])
        s = torch.stack([o[1] for o in out])
        e, a = self.calc_uncertainty(yhat, s)
        return (yhat.mean(dim=0), s.mean(dim=0), e, a)

    def get_metrics(self, x, y, n_samp=50, eps=1e-6):
        """ get uncertainties during training for analysis """
        state = self.training
        self.eval()
        with torch.no_grad():
            yhat, s, ep, al = self.predict(x, n_samp)
            loss = self.criterion((yhat, s), y)
            yhat, s = yhat.detach().cpu(), s.detach().cpu()
            ep, al = ep.detach().cpu(), al.detach().cpu()
            sb = ep / (al + eps)
            eu, au = ep.mean(), al.mean()
            su = sb.mean()
        self.train(state)
        return loss, (yhat, s), (ep, al, sb), (eu, au, su)

    @staticmethod
    def calc_segmentation_uncertainty(logits, sigma):
        mc_prob = F.softmax(logits, dim=2).mean(dim=0)  # channel dim = 2
        epistemic = -1 * (mc_prob * mc_prob.log()).sum(dim=1, keepdim=True).detach().cpu()  # entropy
        aleatoric = sigma.mean(dim=0).mean(dim=1, keepdim=True).detach().cpu()
        return epistemic, aleatoric

    def segment_predict(self, logits, sigma, n_samp=50):
        dist = torch.distributions.Normal(logits, sigma)
        x_hat = dist.rsample((n_samp,)).mean(dim=0)
        _, pred = x_hat.max(dim=1)
        return pred

    def get_segmentation_metrics(self, x, y, n_samp=50, eps=1e-6):
        """ get segmentation uncertainties and other metrics during training for analysis """
        state = self.training
        self.eval()
        with torch.no_grad():
            logits, sigma, ep, al = self.predict(x, n_samp)
            loss = self.criterion((logits, sigma), y)
            sb = ep / (al + eps)
            eu, au = ep.mean(), al.mean()
            su = sb.mean()
            pred = self.segment_predict(logits, sigma, n_samp)
            pred, y = pred.detach().cpu(), y.detach().cpu()
            ds, js = list_to_np((dice(pred, y), jaccard(pred, y)))
        self.train(state)
        return loss, pred, (ep, al, sb), (eu, au, su), (ds, js)

    def fwd_bayesian_segnet(self, x):
        raise NotImplementedError

    def fwd_full_bayesian(self, x):
        raise NotImplementedError


def turn_on_dropout_layers(net):
    def control_func(m):
        classname = m.__class__.__name__
        if 'Dropout' in classname: m.train()
    net.apply(control_func)
