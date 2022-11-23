from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functions.utils import _transpose_and_gather_feat

class FocalLoss(nn.Module):
    '''This is FocalLoss, the same as CornerNet and CenterNet'''
    '''THis gamma, alpha is different from origin FocalLoss'''
    def __init__(self, gamma=2, alpha=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.alpha) # 我感觉这个参数完全没有用，应该是给smooth label准备的

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.gamma) * pos_inds # 这里的power是用作平衡简单困难样本的，相当于FocalLoss里面的gamma
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.gamma) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss
        

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
  
    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class HUModelLoss(nn.Module):
    def __init__(self, opt):
        super(HUModelLoss, self).__init__()
        self.opt = opt

        self.loss_dict = {}
        if not self.opt.two_heads: # and len(self.opt.class_name) == 2
            self.loss_dict['merge'] = {'hm': FocalLoss(), 'wh': RegL1Loss(), 'reg': RegL1Loss()}
        else:
            assert(len(self.opt.class_name) == 2)
            for item in self.opt.class_name:
                self.loss_dict[item] = {'hm': FocalLoss(), 'wh': RegL1Loss(), 'reg': RegL1Loss()}

        self.h_hm_weight = opt.h_hm_weight
        self.p_hm_weight = opt.p_hm_weight
        self.wh_weight = opt.wh_weight
        self.reg_weight = opt.reg_weight

    def forward(self, output, target):
        loss = 0
        hm_loss = 0
        wh_loss = 0
        reg_loss = 0
        head_loss = 0
        person_loss = 0

        assert('merge' in output or 'head' in output or 'person' in output), 'your output type is not right'

        if 'merge' in output:
            hm_loss += self.loss_dict['merge']['hm'](output['merge']['hm'], target['merge']['hm'])
            wh_loss += self.wh_weight * self.loss_dict['merge']['wh'](output['merge']['wh'], target['merge']['reg_mask'], target['merge']['ind'], target['merge']['wh'])
            reg_loss += self.reg_weight * self.loss_dict['merge']['reg'](output['merge']['reg'], target['merge']['reg_mask'], target['merge']['ind'], target['merge']['reg'])   
        else:        
            if 'head' in output:
                hm_loss += self.h_hm_weight * self.loss_dict['head']['hm'](output['head']['hm'], target['head']['hm'])
                wh_loss += self.wh_weight * self.loss_dict['head']['wh'](output['head']['wh'], target['head']['reg_mask'], target['head']['ind'], target['head']['wh'])
                reg_loss += self.reg_weight * self.loss_dict['head']['reg'](output['head']['reg'], target['head']['reg_mask'], target['head']['ind'], target['head']['reg'])
                head_loss = hm_loss + wh_loss + reg_loss
            if 'person' in output:
                hm_loss += self.p_hm_weight * self.loss_dict['person']['hm'](output['person']['hm'], target['person']['hm'])
                wh_loss += self.wh_weight * self.loss_dict['person']['wh'](output['person']['wh'], target['person']['reg_mask'], target['person']['ind'], target['person']['wh'])
                reg_loss += self.reg_weight * self.loss_dict['person']['reg'](output['person']['reg'], target['person']['reg_mask'], target['person']['ind'], target['person']['reg'])
                person_loss = hm_loss + wh_loss + reg_loss                



        loss = hm_loss + wh_loss + reg_loss

        loss_state = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'reg_loss': reg_loss}
        return loss, loss_state