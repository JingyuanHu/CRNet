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
from logger import Logger

#seed = 1027
#torch.manual_seed(seed)  # 为了cpu设置随机种子
#torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子 通过torch.cuda.FloatTensor(1).uniform_()可以查看
#torch.cuda.manual_seed_all(seed) # 为所有gpu设置随机种子 通过torch.cuda.FloatTensor(1).uniform_()可以查看

def main(opt):
    seed = random.randint(1, 1000)
    opt.seed = seed
    # 控制随机种子
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed) 
    torch.cuda.manual_seed_all(opt.seed) 
    cudnn.benchmark = False 
    cudnn.deterministic = True

    ######################################### Log部分没有写，先不处理
    # logger = Logger(opt)
    # save_dir = logger.log_dir


    # Gpu环境设定
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus 
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # 数据集
    train_dataset = TrainValDataset('train', opt, augment=True)
    if opt.train_with_val:
        val_dataset = TrainValDataset('val', opt, augment=False)
    test_dataset = TestDataset('test', opt)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True)
    if opt.train_with_val:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=opt.num_workers,
                                                 pin_memory=True)

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
    
    # dir setting
    if not os.path.isdir(opt.exp_dir):
        os.mkdir(opt.exp_dir)
    
    exp_id = 1
    while(True):
        run_dir = os.path.join(opt.exp_dir, opt.exp_name + '_' + str(exp_id))
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
            break
        else:
            exp_id += 1
    os.mkdir(os.path.join(run_dir, 'checkpoint'))
    os.mkdir(os.path.join(run_dir, 'result'))
    model_dir = os.path.join(run_dir, 'checkpoint')
    result_dir = os.path.join(run_dir, 'result')
    
    # logger
    log = Logger(opt, run_dir)


    # 模型
    model = HUModel(opt)
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Loss 设置
    model_loss = HUModelLoss(opt)

    ## 模型GPU设定
    trainer = Trainer(opt, model, model_loss, optimizer, device)
    start_epoch = 0

    val_best_loss = 10
    #best_map = 0
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        train_output = trainer.train(epoch, train_loader)
        log.write('train epoch {} :'.format(epoch))
        for k, v in train_output.items():
            log.scalar_summary('train_{}'.format(k), v, epoch)
            log.write('| {} {:8f} '.format(k, v))
        log.write('\n')
        if epoch >= 50:
            output_boxes = model_eval(test_loader, device, model, opt)
            _, _, APs, mAP = test_dataset.evaluate_detections(output_boxes, result_dir)
            log.write('test epoch {} : '.format(epoch))
            for index in range(len(opt.class_name)):
                log.write('| AP for {} = {:.4f}'.format(opt.class_name[index], APs[index]))
            log.write('\n')
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            data = {'epoch': epoch, 'state_dict': state_dict}
            torch.save(data, os.path.join(model_dir, 'model_{}_{:5f}.pth'.format(epoch, mAP)))

       
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 'state_dict': state_dict}
        torch.save(data, os.path.join(model_dir, 'model_final.pth'))
 
        if epoch in opt.lr_step:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
               param_group['lr'] = lr
    log.close()






if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
