import argparse
import os
import sys

def parse_opt():
    parser = argparse.ArgumentParser()
    # 学习率
    parser.add_argument('--lr', type=float, default=3e-4, help='Start Learning Rate')
    # 训练epoch
    parser.add_argument('--num_epochs', type=int, default=65, help='Number of train epoch')
    parser.add_argument('--lr_step', default=[50, 60], help='image size of input images')
    
    # 各种权重在loss中的占比
    parser.add_argument('--h_hm_weight', type=float, default=1, help='head heatmap weight')
    parser.add_argument('--p_hm_weight', type=float, default=1, help='person heatmap weight')
    parser.add_argument('--wh_weight', type=float, default=0.1, help='width hegith weight')
    parser.add_argument('--reg_weight', type=float, default=1, help='reg weight')

    # 随机种子
    parser.add_argument('--seed', type=int, default=317, help='random seed')

    # backbone使用
    parser.add_argument('--arch', default='res_50', help='model architecture. Currently tested res_50')

    # num_workers
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader threads. 0 for single-thread.')

    # gpus
    parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')

    # data root path
    parser.add_argument('--rootpath', default='/home/user01/data/HeadDetection/brainwash/VOCShoulderBrainwash', help='data root path')
    #parser.add_argument('--rootpath', default='/home/user01/data/HeadDetection/SCUT_HEAD/VOCSCUT_A', help='data root path')

    # batch_size
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    
    # val
    parser.add_argument('--train_with_val', default=False, type=bool, help='use val dataset when training')

    # two head
    parser.add_argument('--two_heads', default=False, type=bool, help='whether use head and person two heads')

    # cross attention
    parser.add_argument('--correlation_attention', default=False, type=bool, help='whether use correlation attetion')

    # class_name
    parser.add_argument('--class_name', default=['head'], help='the class name using in network, just support head and person')

    # class_weight
    parser.add_argument('--class_weight', default={'person': 0.5, 'head' : 0.5}, help='the class weight using in network, just support head and person')

    # input_size
    parser.add_argument('--input_size', default=512, type=int, help='image size of input images')

    # atss
    parser.add_argument('--atss', default=2, type=int, help='radius when use atss, if <=0, radius use origin gaussian_radius ')

    # dir setting
    parser.add_argument('--exp_dir', default='experiment', help='experiment dir')
    parser.add_argument('--exp_name', default='run', help='exp name')

    # heatmap threshold
    parser.add_argument('--thresh', default=0.0, type=float, help='min score for heatmap nms')

    opt = parser.parse_args()
    return opt
