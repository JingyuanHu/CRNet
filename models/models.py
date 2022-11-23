from networks.resnet import get_backbone_net as get_resnet
from networks.head import get_center_net
import torch.nn as nn

backbone_factory = {'res_50': get_resnet}

head_factory = {'CenterNet': get_center_net}

head_conv_factory = {'res_50': 64}

last_conv_factory = {'res_50': 64}

class HUModel(nn.Module):
    def __init__(self, opt):
        super(HUModel, self).__init__()
        self.opt = opt
        backbone_name = opt.arch
        assert(backbone_name in backbone_factory), 'Your Backbone is not supported'
        self.backbone = backbone_factory[backbone_name]()
        
        last_conv = last_conv_factory[backbone_name]
        head_conv = head_conv_factory[backbone_name]
        self.head = head_factory['CenterNet'](len(opt.class_name), last_conv, head_conv)
        
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
