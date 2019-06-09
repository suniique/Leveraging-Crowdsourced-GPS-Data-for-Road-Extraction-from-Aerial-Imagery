#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(
                num_features=out_channels  #, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(
        self, in_channels, mid_channels, out_channels, stride, dilation, downsample
    ):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1, stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReLU(
            mid_channels, mid_channels, 3, 1, dilation, dilation
        )
        self.increase = _ConvBatchNormReLU(
            mid_channels, out_channels, 1, 1, 0, 1, relu=False
        )
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReLU(
                in_channels, out_channels, 1, stride, 0, 1, relu=False
            )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResBlock(nn.Sequential):
    """Residual Block"""

    def __init__(
        self,
        n_layers,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilation,
        mg=None,
    ):
        super(_ResBlock, self).__init__()

        if mg is None:
            mg = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(mg), "{} values expected, but got: mg={}".format(
                n_layers, mg
            )

        self.add_module(
            "block1",
            _Bottleneck(
                in_channels, mid_channels, out_channels, stride, dilation * mg[0], True
            ),
        )
        for i, g in zip(range(2, n_layers + 1), mg[1:]):
            self.add_module(
                "block" + str(i),
                _Bottleneck(
                    out_channels, mid_channels, out_channels, 1, dilation * g, False
                ),
            )

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d(1)),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear")]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h



class DeepLabV3Plus(nn.Module):
    """DeepLab v3+ (OS=8)"""

    def __init__(self, n_classes, n_blocks=[3,4,15,3], pyramids=[6,12,18], grids=[1,2,4], output_stride=16, num_channels=3):
        super(DeepLabV3Plus, self).__init__()

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        # Encoder
        self.add_module(
            "layer1",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(num_channels, 64, 7, 2, 3, 1)),
                        ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                    ]
                )
            ),
        )
        self.add_module(
            "layer2", _ResBlock(
                n_blocks[0], 64, 64, 256, stride[0], dilation[0])
        )
        self.add_module(
            "layer3", _ResBlock(
                n_blocks[1], 256, 128, 512, stride[1], dilation[1])
        )
        self.add_module(
            "layer4", _ResBlock(
                n_blocks[2], 512, 256, 1024, stride[2], dilation[2])
        )
        self.add_module(
            "layer5",
            _ResBlock(n_blocks[3], 1024, 512, 2048,
                      stride[3], dilation[3], mg=grids),
        )
        self.add_module("aspp", _ASPPModule(2048, 256, pyramids))
        self.add_module(
            "fc1", _ConvBatchNormReLU(
                256 * (len(pyramids) + 2), 256, 1, 1, 0, 1)
        )
        # Decoder
        self.add_module("reduce", _ConvBatchNormReLU(256, 48, 1, 1, 0, 1))
        self.add_module(
            "fc2",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(304, 256, 3, 1, 1, 1)),
                        ("conv2", _ConvBatchNormReLU(256, 256, 3, 1, 1, 1)),
                        ("conv3", nn.Conv2d(256, n_classes, kernel_size=1)),
                    ]
                )
            ),
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h_ = self.reduce(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)
        h = self.fc1(h)
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear")
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear")
        h = F.sigmoid(h)
        return h

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()


if __name__ == "__main__":
    model = DeepLabV3Plus(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        pyramids=[6, 12, 18],
        grids=[1, 2, 4],
        output_stride=16,
    )
    model.freeze_bn()
    model.eval()
    print(list(model.named_children()))
    image = torch.randn(1, 3, 513, 513)
    print(model(image)[0].size())
