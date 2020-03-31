#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
okulonet

okulonet (2d deeplabv3-derived model) for synthesis or segmentation

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Feb. 25, 2019
"""

__all__ = ['OkuloNet']

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models import resnet

from ..learn import *
from ..util import *
from .unet_tools import *

model_urls = {
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


class OkuloNet(nn.Module):
    def __init__(self, ic=1, oc=1, p=0., beta=25., bayesian=False, laplacian=True, segmentation=None,
                 fc=8, sc=3):
        super().__init__()
        model = deeplabv3_resnet101(pretrained=True)
        self.update_conv1 = sc != 3
        self.backbone = model.backbone
        if self.update_conv1:
            self.backbone.conv1 = nn.Conv2d(sc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.head = model.classifier
        self.head[0].project[3] = nn.Dropout2d(p)  # change dropout to channel dropout
        self.head[4] = nn.Sequential(*conv2d(256, 256, 3))  # change last layer to not be classifier
        self.start = unet_block2d(ic, fc, sc, 7, 5)
        self.up5 = unet_block2d(256+2048, 512, 512, 3, 3)
        self.up4 = unet_block2d(512+1024, 256, 256, 3, 3)
        self.up3 = unet_block2d(256+512, 128, 128, 3, 3)
        self.up2 = unet_block2d(128+256, 64, 64, 3, 3)
        self.up1 = unet_block2d(64+64, 32, 32, 3, 3)
        self.end = unet_block2d(32+sc, 32, 32, 3, 3)
        self.syn = nn.Sequential(*conv2d(32, 32, 3), nn.Conv2d(32, oc, 1))
        self.unc = nn.Sequential(*conv2d(32, 32, 3), nn.Conv2d(32, oc, 1))
        self.p = p
        self.bayesian = bayesian
        self.segmentation = segmentation
        if bayesian:
            if segmentation is not None:
                self.criterion = ExtendedCrossEntropy(beta, **segmentation)
            else:
                self.criterion = LaplacianDiagLoss(beta) if laplacian else GaussianDiagLoss(beta)
        else:
            if segmentation is not None:
                self.criterion = BinaryFocalLoss(beta, **segmentation)
            else:
                self.criterion = L1OnlyLoss(beta) if laplacian else MSEOnlyLoss(beta)

    def freeze_full(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False

    def freeze(self):
        if not self.update_conv1:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        for param in self.backbone.layer3.parameters():
            param.requires_grad = False

    def dropout(self,x):
        use_dropout = self.bayesian or self.training
        return F.dropout2d(x, self.p, training=use_dropout, inplace=False)

    @staticmethod
    def cat(x,r): return torch.cat((x, r), dim=1)

    @staticmethod
    def interp(x,s): return F.interpolate(x, size=s, mode='bilinear', align_corners=True)

    def interpcat(self,x,r): return self.cat(self.interp(x, r.shape[2:]), r)

    def forward(self, x):
        x = self.start(x)
        orig = x.clone()
        features = self.backbone(x)
        x = features["out"]
        x = self.head(x)
        x = self.interpcat(x, features['out'])
        x = self.up5(x)
        x = self.dropout(x)
        x = self.interpcat(x, features['mid3'])
        x = self.up4(x)
        x = self.dropout(x)
        x = self.interpcat(x, features['mid2'])
        x = self.up3(x)
        x = self.dropout(x)
        x = self.interpcat(x, features['mid1'])
        x = self.up2(x)
        x = self.dropout(x)
        x = self.interpcat(x, features['start'])
        x = self.up1(x)
        x = self.dropout(x)
        x = self.interpcat(x, orig)
        x = self.end(x)
        yhat, s = self.syn(x), self.unc(x)
        return yhat, s

    def binary_segmentation_uncertainty_predict(self, x, n_samp=50, eps=1e-6):
        logits, sigmas = [], []
        for _ in range(n_samp):
            logit, sigma = self.forward(x)
            logits.append(logit.detach().cpu())
            sigmas.append(sigma.detach().cpu())
        logits = torch.stack(logits)
        epistemic = logits.var(dim=0, unbiased=True)
        logit = logits.mean(dim=0)
        probit = torch.sigmoid(logit)
        entropy = -1 * (probit * (probit + eps).log() + ((1 - probit) * (1 - probit + eps).log()))  # entropy
        sigma = torch.stack(sigmas).mean(dim=0)
        aleatoric = F.softplus(sigma)
        return (logit, sigma, epistemic, entropy, aleatoric)

    def get_binary_segmentation_metrics(self, x, y, n_samp=50, eps=1e-6):
        """ get segmentation uncertainties and other metrics during training for analysis """
        state = self.training
        self.eval()
        with torch.no_grad():
            y = y.detach().cpu()
            logit, sigma, ep, en, al = self.binary_segmentation_uncertainty_predict(x, n_samp, eps)
            if self.criterion.weight is not None:
                device = self.criterion.weight.device
                self.criterion.weight = self.criterion.weight.cpu()
            loss = self.criterion((logit, sigma), y)
            sb = ep / (al + eps)
            eu, au = ep.mean(), al.mean()
            nu = en.mean()
            su = sb.mean()
            pred = (logit >= 0)
            ds, js = list_to_np((dice(pred, y), jaccard(pred, y)))
        self.train(state)
        if self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(device)
        return loss, pred, (ep, en, al, sb), (eu, nu, au, su), (ds, js)


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])
    return_layers = {'layer4': 'out',   # 2048 channels
                     'layer3': 'mid3',  # 1024 channels
                     'layer2': 'mid2',  # 512 channels
                     'layer1': 'mid1',  # 256 channels
                     'relu': 'start'}   # 64 channels
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)
    model_map = {'deeplabv3': (DeepLabHead, DeepLabV3)}
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]
    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs):
    if pretrained:
        aux_loss = True
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)
    return model


def deeplabv3_resnet50(pretrained=False, progress=True,
                       num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet50', pretrained, progress, num_classes, aux_loss, **kwargs)


def deeplabv3_resnet101(pretrained=False, progress=True,
                        num_classes=21, aux_loss=None, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, aux_loss, **kwargs)
