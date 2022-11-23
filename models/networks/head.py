from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .aspp import ASPP
from .gcn import SNLblock2d

class NormalCenterNet(nn.Module):
    def __init__(self, num_class, last_conv, head_conv):
        super(NormalCenterNet, self).__init__()
        assert(num_class == 1 or num_class == 2)
        self.hm = nn.Sequential(#nn.Conv2d(last_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                #  nn.ReLU(inplace=True),
                                  nn.Conv2d(head_conv, num_class, kernel_size=1, stride=1, padding=0)
                                )
        self.wh = nn.Sequential(#nn.Conv2d(last_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                #nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0)
                                )
        self.reg =nn.Sequential(#nn.Conv2d(last_conv, head_conv, kernel_size=3, padding=1, bias=True),
                                #nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0)
                                )
        #self.gcn = SNLblock2d(last_conv, last_conv)
        self.aspp  = nn.Sequential(ASPP(64, 64, [6, 12, 18, 24]), nn.ReLU(inplace=True))

    def forward(self, x):
        #x = self.gcn(x
        x_ = self.aspp(x)

        x = x_ + x
        
        hm = self.hm(x)
        hm = torch.clamp(hm.sigmoid_(), min=1e-6, max=1-1e-6)
        wh = self.wh(x)
        reg = self.reg(x)
        z = {}
        z['merge'] = {'hm': hm, 'wh': wh, 'reg': reg}
        return z

    def init_weights(self):
        for i, m in enumerate(self.hm.modules()):
            if isinstance(m, nn.Conv2d):
                #nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, -3)

        for i, m in enumerate(self.wh.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

        for i, m in enumerate(self.reg.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)



def get_center_net(num_class, last_conv=256, head_conv=256):
    head_part = NormalCenterNet(num_class, last_conv, head_conv)
    head_part.init_weights()
    return head_part

