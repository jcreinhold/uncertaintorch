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
from .unet_tools import *

model_urls = {
    'deeplabv3_resnet50_coco': None,
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}


class OkuloNet(nn.Module):
    def __init__(self, ic=1, oc=1, p=0., beta=25., bayesian=False, laplacian=True, segmentation=None):
        super().__init__()
        model = deeplabv3_resnet101(pretrained=True)
        self.backbone = model.backbone
        self.head = model.classifier
        self._freeze()
        del self.backbone.fc, self.backbone.avgpool
        self.head[0].project[3] = nn.Dropout2d(p)  # change dropout to channel dropout
        self.start = unet_block2d(ic, 8, 3, 5, 3)
        self.up5 = unet_block2d(21+2048, 1024, 1024, 5, 3)
        self.up4 = unet_block2d(1024+1024, 1024, 1024, 5, 3)
        self.up3 = unet_block2d(1024+512, 512, 512, 5, 3)
        self.up2 = unet_block2d(512+256, 256, 256, 5, 3)
        self.up1 = unet_block2d(256+64, 64, 64, 5, 3)
        self.end = unet_block2d(64+3, 32, 32, 5, 3)
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

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
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