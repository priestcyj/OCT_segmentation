import os
import json
import platform

import numpy as np
from utils import show_tensor_img
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import cv2
from PIL import Image
import torch.utils.data
import torch.nn.functional as F
from sdf import SDFCalculator


def make_tran(SIZE_IMAGEH, SIZE_IMAGEW, mode):
    ALB_TWIST = []
    if mode == 'train':
        ALB_TWIST = alb.Compose([
            alb.PadIfNeeded(min_height=SIZE_IMAGEH, min_width=SIZE_IMAGEW, p=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            alb.CropNonEmptyMaskIfExists(height=SIZE_IMAGEH, width=SIZE_IMAGEW, p=1),
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomBrightnessContrast(p=1),
            ToTensorV2()
        ])
    elif mode == 'val':
        ALB_TWIST = ToTensorV2()
        # ALB_TWIST = alb.Compose([
        #     alb.HorizontalFlip(p=1),
        #     alb.VerticalFlip(p=0.5),
        #     ToTensorV2()
        # ])
    elif mode == 'test':
        ALB_TWIST = ToTensorV2()
    return ALB_TWIST



class HEG(torch.utils.data.Dataset):
    def __init__(self, config, img_ids, num_classes, lens, mode=None):
        self.config = config
        self.img_ids = img_ids
        if platform.system() == 'Windows':
            self.data_dir = r'D:\My_Project\yf_code\OCT_DATA\HEG_Processed_by_Dr_Fan_Yang'
        else:
            self.data_dir = r'/data/yf/OCT_DATA/HEG_Processed_by_Dr_Fan_Yang'
        self.num_classes = num_classes
        self.mode = mode
        self.img_dir = os.path.join(self.data_dir, 'image')
        self.lab_dir = os.path.join(self.data_dir, 'label')
        self.edge_lab_dir = os.path.join(self.data_dir, 'edge_label')

        # if mode != 'test':
        #     self.img_dir = self.img_dir.replace('image', 'slice_image')
        #     self.lab_dir = self.lab_dir.replace('label', 'slice_label')
        #     self.edge_lab_dir = self.edge_lab_dir.replace('edge_label', 'slice_edge_label')

        self.transform = make_tran(256, 256, mode)

        self.lens = lens
        self.exeNums = {'train': max(1, 735 // self.lens['train']), 'val': 1, 'test': 1}

    def __len__(self):
        if self.mode == 'test':
            return self.lens['test']
        elif self.mode == 'val':
            return self.lens['val']
        return self.lens['train'] * self.exeNums[self.mode]

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx = idx % self.lens['test']
        elif self.mode == 'val':
            idx = idx % self.lens['val']
        elif self.mode == 'train':
            idx = idx % self.lens['train']

        img_id = self.img_ids[idx % self.lens['train']]

        img = np.array(Image.open(os.path.join(self.img_dir, img_id)).convert('L'))
        lab = np.array(Image.open(os.path.join(self.lab_dir, img_id)).convert('P'))
        edge_lab = np.array(Image.open(os.path.join(self.edge_lab_dir, img_id)).convert('P'))

        # show_tensor_img(img, 'img', 800)
        original_label = lab


        # 数据增强
        data = {'image': img, 'masks': [lab, edge_lab]}
        pics = self.transform(**data)
        img = pics['image'].float() / 255
        lab = pics['masks'][0].long()
        edge_lab = pics['masks'][1].long()

        # 处理sdf_label
        # def edge_to_seg(edge):
        #     C = torch.unique(edge).shape[0]
        #     prob = torch.zeros_like(edge)
        #     one_hot_edge = F.one_hot(edge)
        #     for c in range(1, C-1):
        #         this_layer = torch.argmax(one_hot_edge[:, :, c], dim=0)
        #         next_layer = torch.argmax(one_hot_edge[:, :, c+1], dim=0)
        #         for col in range(this_layer.shape[0]):
        #             this_idx = this_layer[col]
        #             next_idx = next_layer[col]
        #             if this_idx < next_idx:
        #                 prob[this_idx:next_idx, col] = c
        #             elif (this_idx == 0) or (next_idx == 0) :
        #                 continue
        #             else:
        #                 prob[next_idx:this_idx, col] = c
        #     return prob


        # prob_labels = edge_to_seg(edge_lab)
        prob_labels = F.one_hot(lab, num_classes=self.config['num_classes']).permute((2, 0, 1)).float()
        sdf_calculator = SDFCalculator(self.config, max_workers=4)  # 设置最大工作线程数
        sdf_label = sdf_calculator.compute_sdf(prob_labels)
        sdf_label = sdf_label

        # show_tensor_img(img, 'img', 800)
        return img, {'label': lab, 'edge_label':edge_lab, 'sdf_label': sdf_label}, {'img_id': os.path.splitext(img_id)[0]}










