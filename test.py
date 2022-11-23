import torch
import torch.nn as nn
import numpy as np
import argparse
import yaml
import os
import random 

from models.models import HUModel
from datasets.datasets import TrainValDataset, TestDataset
from models.loss import HUModelLoss
from torch.backends import cudnn
from trains.trainer import Trainer
from trains.eval import model_eval
from opts import parse_opt

def main(opt):
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed) 
    torch.cuda.manual_seed_all(opt.seed) 
    cudnn.benchmark = False 
    cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus 
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')    
    
    test_dataset = TestDataset('test', opt)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=opt.num_workers,
                                             pin_memory=True)
    # remove annotation_cache prevent error
    cachefile = os.path.join(os.path.join(opt.rootpath, 'annotations_cache'), 'annots.pkl')
    if os.path.isfile(cachefile):
        try:
            os.remove(cachefile)
        except BaseException as e:
            print(e)
    
    model = HUModel(opt)
    model = model.to(device)
    output_boxes = model_eval(test_loader, device, model, opt, model_path='experiment/run_12/checkpoint/model_best.pth')
    #print(output_boxes)
    rec, prec, APs, mAP = test_dataset.evaluate_detections(output_boxes, '/home/user01/HUDetect/result/')
    
    #for i, cls in enumerate(opt.class_name):
    #    ap = APs[i]
    #    print('AP for {} = {:.4f}'.format(cls, ap))
    #print('Mean AP = {:.4f}'.format(mAP))
    #print(prec)
    #print(rec)
    #print('Precision = {:.4f}'.format(prec))
    #print('Recall = {:.4f}'.format(rec))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
