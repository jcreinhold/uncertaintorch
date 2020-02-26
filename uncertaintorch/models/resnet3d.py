#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resnet3d

3d resnet model (backbone for deeplabv3)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: December 31, 2019
"""

__all__ = ['ResNet3d', 'resnet3d18', 'resnet3d34', 'resnet3d50', 'resnet3d101']

import torch
from torch import nn
import torch.nn.functional as F

BASE_WIDTH = 32
EXPANSION = 2


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.ReplicationPad3d(dilation),
                         nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   groups=groups, bias=False, dilation=dilation))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = EXPANSION
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=BASE_WIDTH, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / BASE_WIDTH)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=BASE_WIDTH, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=1, dropout_rate=0., bayesian=False):
        super(ResNet3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.p = dropout_rate
        self.bayesian = bayesian
        self.inplanes = BASE_WIDTH
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(nn.ReplicationPad3d(3),
                                   nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, bias=False))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, BASE_WIDTH, layers[0])
        self.layer2 = self._make_layer(block, BASE_WIDTH*2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, BASE_WIDTH*(2 if replace_stride_with_dilation[0] else 4),
                                       layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, BASE_WIDTH*(2 if replace_stride_with_dilation[1] else 8),
                                       layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def dropout(self, x):
        use_dropout = self.training or self.bayesian
        return F.dropout3d(x, self.p, training=use_dropout, inplace=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        start = x.clone()
        x = self.maxpool(x)
        x = self.layer1(x)
        mid = x.clone()
        x = self.dropout(x)
        x = self.dropout(self.layer2(x))
        x = self.dropout(self.layer3(x))
        x = self.dropout(self.layer4(x))
        return x, start, mid

    def forward(self, x):
        return self._forward_impl(x)


def resnet3d18(**kwargs):
    return ResNet3d(Bottleneck, [2, 2, 2, 2], **kwargs)


def resnet3d34(**kwargs):
    return ResNet3d(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet3d50(**kwargs):
    return ResNet3d(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet3d101(**kwargs):
    return ResNet3d(Bottleneck, [3, 4, 23, 3], **kwargs)

