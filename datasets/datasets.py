import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F 
from torch.utils.data import Dataset 
from PIL import Image
import xml.etree.ElementTree as ET
from utils import image_trans as img_trans
from utils import utils as utils
import math
from .voc_eval import voc_eval
import pickle


def get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

Support_class = ['head', 'person']

class TrainValDataset(Dataset):
    def __init__(self, split, opt, augment):
        for item in opt.class_name:
            assert(item in Support_class), 'this class is not supported, try head or person'
        
        # 类别名定义一个index作为后面索引的重要依据    
        self.cat_ids = {v: i for i, v in enumerate(opt.class_name)}
        
        self.rootpath = opt.rootpath
        self.annot_path = os.path.join('%s', 'Annotations', '%s.xml')
        self.img_path = os.path.join('%s', 'JPEGImages', '%s.jpg')
        
        self.sample_names = list()
        for line in open(os.path.join(self.rootpath, 'ImageSets', 'Main', split + '.txt')):
            self.sample_names.append(line.strip())
        self.num_samples = len(self.sample_names)
        
        self.input_size = opt.input_size

        self.augment = augment

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self.eig_vec = np.array([[-0.58752847, -0.69563484, 0.41340352],
                                  [-0.5832747, 0.00994535, -0.81221408],
                                  [-0.56089297, 0.71832671, 0.41158938]
                                 ], dtype=np.float32)
        
        self.opt = opt
        self.max_objs = 100

        self.split = split
        
        for item in self.opt.class_name:
            assert(item in ['head', 'person']), 'Dont support ' + item + ' right now'
        assert(len(self.opt.class_name) > 0), 'class_name must larger than 0'
        

    def __len__(self):
        return self.num_samples    
    
    def __getitem__(self, index):
        sample_name = self.sample_names[index]
        tree = ET.parse(self.annot_path % (self.rootpath, sample_name)).getroot()
        img = cv2.imread(self.img_path % (self.rootpath, sample_name))
        
        anns =  tree.findall('object')
        num_objs = len(anns)
        
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)
        
        scale = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.input_size, self.input_size

        flipped = False
        if self.augment:
            scale = scale * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)
            
            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] =  width - center[0] - 1
        
        # 计算随机crop后的透视矩阵    
        trans_input = img_trans.get_affine_transform(center, scale, 0, [input_w, input_h])
        
        # 得到随机crop后并进行透视变换的图像
        trans_img = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        
        trans_img = (trans_img.astype(np.float32) / 255.)
        
        if self.augment:
            img_trans.color_aug(self.data_rng, trans_img, self.eig_val, self.eig_vec)
        
        trans_img = (trans_img - self.mean) / self.std
        
        trans_img = trans_img.transpose(2, 0, 1)

        output_h = input_h // 4
        output_w = input_w // 4
        
        trans_output = img_trans.get_affine_transform(center, scale, 0, [output_w, output_h])

        draw_gaussian = utils.draw_umich_gaussian
        
        output_dict = {}
        output_dict['input'] = trans_img
        for class_name in self.opt.class_name:
            hm = np.zeros((1, output_h, output_w), dtype=np.float32)
            wh = np.zeros((self.max_objs // len(self.opt.class_name), 2), dtype=np.float32)
            reg = np.zeros((self.max_objs // len(self.opt.class_name), 2), dtype=np.float32)
            ind = np.zeros((self.max_objs // len(self.opt.class_name)), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs // len(self.opt.class_name)), dtype=np.uint8)

            output_dict[class_name] = {'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask, 'obj_count': 0}
        
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            class_name = ann.find('name').text
            if class_name not in self.opt.class_name:
                continue
            bbox = ann.find('bndbox')
            bbox = np.array([float(bbox.find('xmin').text), float(bbox.find('ymin').text), float(bbox.find('xmax').text), float(bbox.find('ymax').text)])
            cls_id = int(self.cat_ids[class_name])
            
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = img_trans.affine_transform(bbox[:2], trans_output)
            bbox[2:] = img_trans.affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            
            if h > 0 and w > 0:
                if self.opt.atss <= 0:
                    radius = utils.gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                else:
                    radius = self.opt.atss
                
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(output_dict[class_name]['hm'][0], ct_int, radius)
                output_dict[class_name]['wh'][output_dict[class_name]['obj_count']] = 1. * w, 1. * h
                output_dict[class_name]['ind'][output_dict[class_name]['obj_count']] = ct_int[1] * output_w + ct_int[0]
                output_dict[class_name]['reg'][output_dict[class_name]['obj_count']] = ct - ct_int
                output_dict[class_name]['reg_mask'][output_dict[class_name]['obj_count']] = 1
                output_dict[class_name]['obj_count'] += 1
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        

        #if not self.split == 'train':
        if not ('train' in self.split):
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else np.zeros((1, 6), dtype=np.float32)
            meta = {'c': center, 's': scale, 'gt_det': gt_det, 'img_name': sample_name}
            output_dict['meta'] = meta

        
        # 看看是否需要merge output
        if self.opt.two_heads:
            assert(len(self.opt.class_name)== 2), 'Must use two class to match two heads'
        else:
            if len(self.opt.class_name)== 2:  # merge hm,wh,reg,ind,reg_mask and so on
                hm = np.zeros((2, output_h, output_w), dtype=np.float32)
                wh = np.zeros((self.max_objs, 2), dtype=np.float32)
                reg = np.zeros((self.max_objs, 2), dtype=np.float32)
                ind = np.zeros((self.max_objs), dtype=np.int64)
                reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

                hm[0, :, :] = output_dict[self.opt.class_name[0]]['hm']
                hm[1, :, :] = output_dict[self.opt.class_name[1]]['hm']

                wh[:self.max_objs // len(self.opt.class_name), :] = output_dict[self.opt.class_name[0]]['wh']
                wh[self.max_objs // len(self.opt.class_name):, :] = output_dict[self.opt.class_name[1]]['wh']
                
                reg[:self.max_objs // len(self.opt.class_name), :] = output_dict[self.opt.class_name[0]]['reg']
                reg[self.max_objs // len(self.opt.class_name):, :] = output_dict[self.opt.class_name[1]]['reg']

                ind[:self.max_objs // len(self.opt.class_name)] = output_dict[self.opt.class_name[0]]['ind']
                ind[self.max_objs // len(self.opt.class_name):] = output_dict[self.opt.class_name[1]]['ind']

                reg_mask[:self.max_objs // len(self.opt.class_name)] = output_dict[self.opt.class_name[0]]['reg_mask']
                reg_mask[self.max_objs // len(self.opt.class_name):] = output_dict[self.opt.class_name[1]]['reg_mask']

                obj_count = output_dict[self.opt.class_name[0]]['obj_count'] + output_dict[self.opt.class_name[1]]['obj_count']

                output_dict['merge'] = {'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask, 'obj_count': obj_count}
            else:
                hm = np.zeros((1, output_h, output_w), dtype=np.float32)
                wh = np.zeros((self.max_objs, 2), dtype=np.float32)
                reg = np.zeros((self.max_objs, 2), dtype=np.float32)
                ind = np.zeros((self.max_objs), dtype=np.int64)
                reg_mask = np.zeros((self.max_objs), dtype=np.uint8)                
                hm[0, :, :] = output_dict[self.opt.class_name[0]]['hm']

                wh[:self.max_objs // len(self.opt.class_name), :] = output_dict[self.opt.class_name[0]]['wh']
                
                reg[:self.max_objs // len(self.opt.class_name), :] = output_dict[self.opt.class_name[0]]['reg']

                ind[:self.max_objs // len(self.opt.class_name)] = output_dict[self.opt.class_name[0]]['ind']

                reg_mask[:self.max_objs // len(self.opt.class_name)] = output_dict[self.opt.class_name[0]]['reg_mask']

                obj_count = output_dict[self.opt.class_name[0]]['obj_count']

                output_dict['merge'] = {'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask, 'obj_count': obj_count}
        return output_dict
                
class TestDataset(Dataset):
    def __init__(self, split, opt):
        for item in opt.class_name:
            assert(item in Support_class), 'this class is not supported, try head or person'
        
        # 类别名定义一个index作为后面索引的重要依据    
        self.cat_ids = {v: i for i, v in enumerate(opt.class_name)}
        
        self.rootpath = opt.rootpath
        self.annot_path = os.path.join('%s', 'Annotations', '%s.xml')
        self.img_path = os.path.join('%s', 'JPEGImages', '%s.jpg')
        
        self.sample_names = list()
        for line in open(os.path.join(self.rootpath, 'ImageSets', 'Main', split + '.txt')):
            self.sample_names.append(line.strip())
        self.num_samples = len(self.sample_names)
        
        self.input_size = opt.input_size

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        
        self.opt = opt
        self.max_objs = 500

        self.split = split

        for item in self.opt.class_name:
            assert(item in ['head', 'person']), 'Dont support ' + item + ' right now'
        assert(len(self.opt.class_name) > 0), 'class_name must larger than 0'

    def __len__(self):
        return self.num_samples    
    
    def __getitem__(self, index):
        sample_name = self.sample_names[index]
        image = cv2.imread(self.img_path % (self.rootpath, sample_name))

        height, width = image.shape[0: 2]
        center = np.array([width // 2, height // 2], dtype=np.float32)

        scale = max(height, width) * 1.0
        inp_height, inp_width = self.input_size, self.input_size        

        trans_input = img_trans.get_affine_transform(center, scale, 0, [inp_width, inp_height])

        inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height), flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1)
        images = torch.from_numpy(images)

        meta = {'input': images, 'center': center, 'scale': scale, 'out_height': inp_height // 4, 'out_width': inp_width // 4}
        return meta

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes, output_dir)
        rec, prec, aps, map = self._do_python_eval(output_dir)
        return rec, prec, aps, map

    def _get_voc_results_file_template(self, output_dir):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(output_dir)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
    
    def _write_voc_results_file(self, all_boxes, output_dir):
        class_name = ['__background__'] + self.opt.class_name
        for cls_ind, cls in enumerate(class_name):
            cls_ind = cls_ind
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.sample_names):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index, dets[k, -1],dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        name = self.split
        annopath = os.path.join(self.rootpath, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.rootpath, 'ImageSets','Main', name + '.txt')
        cachedir = os.path.join(self.rootpath, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        #use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = False
        #print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        class_name = ['__background__'] + self.opt.class_name
        for i, cls in enumerate(class_name):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        #print('Results:')
        #for ap in aps:
        #    print('{:.3f}'.format(ap))
        #print('{:.3f}'.format(np.mean(aps)))
        #print('~~~~~~~~')
        #print('')
        #print('--------------------------------------------------------------')
        #print('Results computed with the **unofficial** Python eval code.')
        #print('Results should be very close to the official MATLAB eval code.')
        #print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        #print('-- Thanks, The Management')
        #print('--------------------------------------------------------------')
        return rec, prec, aps, np.mean(aps)




