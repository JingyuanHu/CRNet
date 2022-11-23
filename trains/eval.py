from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
from progress.bar import Bar
from functions.utils import _transpose_and_gather_feat, _gather_feat
#from utils import utils as utils
from tqdm import tqdm
import numpy as np
import math
from utils import image_trans as img_trans
from torchvision.ops import nms

def post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = img_trans.transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = img_trans.transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([dets[i, inds, :4].astype(np.float32),
                                               dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
        ret.append(top_preds)
    return ret


def heat_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    
    topk_inds = topk_inds % (height * width)
    
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    
    topk_clses = (topk_ind / K).int()
    
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs



def decode(heat, wh, reg, thresh, add_index=0, K=100):
    batch, _, height, width = heat.size()
    heat = heat_nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    clses += add_index

    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]   

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2).detach().cpu().numpy() 

    keep_inds = detections[:, :, 4] > thresh
    detections = np.array([detections[keep_inds]])

    return detections




@torch.no_grad()
def model_eval(dataloader, device, model, opt, model_path=None):
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        # check loaded parameters and created model parameters
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]

        for k in model_state_dict:
            if not (k in state_dict):
                state_dict[k] = model_state_dict[k]

        model.load_state_dict(state_dict, strict=False) 
    model.eval()

    s = 'Test: '

    all_boxes = [[[] for _ in range(len(dataloader))] for _ in range(len(opt.class_name) + 1)]
    for batch_i, batch in enumerate(tqdm(dataloader, desc=s)):
        img = batch['input']

        # RUN
        img = img.to(device=device, non_blocking=True)
        out = model(img)
        decode_out = []
        if 'merge' in out:
            decode_out.append(decode(out['merge']['hm'], out['merge']['wh'], out['merge']['reg'], opt.thresh))   # bboxes, scores, clses
        else:
            for name_index in range(len(opt.class_name)):
                name = opt.class_name[name_index]
                decode_out.append(decode(out[name]['hm'], out[name]['wh'], out[name]['reg'], opt.thresh, add_index=name_index))

        decode_out = np.array(decode_out)
        decode_out = decode_out.reshape(1, -1 , 6)

        dets = post_process(decode_out, batch['center'].numpy(), batch['scale'].numpy(), batch['out_height'].numpy()[0], batch['out_width'].numpy()[0], len(opt.class_name))

        for j in range(1, len(opt.class_name) + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        
        detections = []
        detections.append(dets[0])
        results = {}
        for j in range(1, len(opt.class_name) + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
            #soft_nms(results[j], Nt=0.5, method=2)

        #boxes, scores, iou_threshold
        scores_ = results[1][:, 4]
        boxes_ = results[1][:, :4]
        output_index= nms(torch.tensor(boxes_), torch.tensor(scores_), 0.7)
        results[1] = results[1][output_index]

        scores = np.hstack([results[j][:, 4] for j in range(1, len(opt.class_name) + 1)])
        if len(scores) > 100:
            kth = len(scores) - 100
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, len(opt.class_name) + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        for j in range(1, len(opt.class_name) + 1):
            all_boxes[j][batch_i] = results[j]

    return all_boxes









