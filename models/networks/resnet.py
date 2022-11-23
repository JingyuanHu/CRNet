# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .gcn import SNLblock2d
from .aspp import ASPP

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1 # 输出通道倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Conv1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # Conv2
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # Downsample
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4 # 输出通道倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # conv1 1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # conv2 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # conv3 1x1
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False

        super(ResNet, self).__init__()
        # conv1_x
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # deconv
        #self.deconv_layers = self._make_deconv_layer(
        #    [256, 256, 256],
        #    [4, 4, 4],
        #)

        # deconv
        #self.deconv_layers = self._make_deconv_layer(
        #    [256, 128, 64],
        #    [4, 4, 4],
        #)


        #self.conv4_0 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_0 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_0 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_0 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv4_1x1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.snl_4 = SNLblock2d(512, 512)

        self.conv3_1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.snl_3 = SNLblock2d(256, 256)

        self.conv2_1x1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.snl_2 = SNLblock2d(128, 128)

        self.conv1_1x1 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.snl_1 = SNLblock2d(64, 64)


        self.deconv_4_3 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
                                        nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True)])
        self.deconv_3_2 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
                                        nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True)])
        self.deconv_2_1 = nn.Sequential(*[nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
                                        nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True)])

        self.out_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        #self.aspp = ASPP(64, 64, [3, 5, 9, 17])


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels):
        assert len(num_filters) == len(num_filters), \
        'ERROR: len(num_filters) is different len(num_filters)'

        num_layers = len(num_filters)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        layer1_x = self.layer1(x)
        layer2_x = self.layer2(layer1_x)
        layer3_x = self.layer3(layer2_x)
        layer4_x = self.layer4(layer3_x)

        #layer4_x = self.conv4_0(layer4_x)
        layer3_x = self.conv3_0(layer3_x)
        layer2_x = self.conv2_0(layer2_x)
        layer1_x = self.conv1_0(layer1_x)

        layer4_x = self.conv4_1x1(layer4_x)
        layer4_x = self.snl_4(layer4_x)

        layer3_x = layer3_x + self.deconv_4_3(layer4_x)

        layer3_x = self.conv3_1x1(layer3_x)
        layer3_x = self.snl_3(layer3_x)

        layer2_x = layer2_x + self.deconv_3_2(layer3_x)

        layer2_x = self.conv2_1x1(layer2_x)
        layer2_x = self.snl_2(layer2_x)

        layer1_x = layer1_x + self.deconv_2_1(layer2_x)
        layer1_x = self.conv1_1x1(layer1_x)
        layer1_x = self.snl_1(layer1_x)

        x = self.out_conv(layer1_x)

        return x


    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            for deconv in [self.deconv_4_3, self.deconv_3_2, self.deconv_2_1]:
                for _, m in deconv.named_modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        nn.init.normal_(m.weight, std=0.001)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            for conv in [self.conv4_1x1, self.conv3_1x1, self.conv1_1x1, self.conv1_1x1, self.out_conv]:
                nn.init.normal_(conv.weight, std=0.001)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            #print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_backbone_net(num_layers=50):
    block_class, layers = resnet_spec[num_layers]
    model = ResNet(block_class, layers)
    model.init_weights(num_layers, pretrained=True)
    return model

    # input 1x3x300x300
    # output 1x2048x10x10
