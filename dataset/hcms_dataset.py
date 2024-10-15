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
        ALB_TWIST = alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            ToTensorV2()
        ])
    elif mode == 'test':
        ALB_TWIST = ToTensorV2()
    return ALB_TWIST



class HCMS(torch.utils.data.Dataset):
    def __init__(self, config, img_ids, num_classes, lens, mode=None):
        self.config = config
        self.img_ids = img_ids
        if platform.system() == 'Windows':
            self.data_dir = r'D:\My_Project\yf_code\OCT_DATA\HCMS_Processed_by_Dr_Fan_Yang'
        else:
            self.data_dir = r'/data/yf/OCT_DATA/HCMS_Processed_by_Dr_Fan_Yang'
        self.num_classes = num_classes
        self.mode = mode
        self.img_dir = os.path.join(self.data_dir, 'image')
        self.lab_dir = os.path.join(self.data_dir, 'label')
        self.edge_lab_dir = os.path.join(self.data_dir, 'edge_label')
        self.sdf_lab_dir = os.path.join(self.data_dir, 'sdf_label')

        self.prep_tran = alb.Resize(p=1, height=256, width=512, interpolation=cv2.INTER_NEAREST)
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
        img_id = self.img_ids[idx]
        # img = cv2.imread(os.path.join(self.img_dir, img_id), cv2.IMREAD_GRAYSCALE)
        # label = cv2.imread(os.path.join(self.lab_dir, img_id), cv2.IMREAD_GRAYSCALE)
        img = np.array(Image.open(os.path.join(self.img_dir, img_id)).convert('L'))
        lab = np.array(Image.open(os.path.join(self.lab_dir, img_id)).convert('P'))
        edge_lab = np.array(Image.open(os.path.join(self.edge_lab_dir, img_id)).convert('P'))


        # bds_path = os.path.join(self.lab_dir, img_id.replace('.png', '.txt'))
        # with open(bds_path, 'r') as file:
        #     bds_list = json.load(file)['bds']
        #     bds = np.array(bds_list)

        # 数据增强
        img = self.prep_tran(image=img)['image']
        lab = self.prep_tran(image=lab)['image']

        edge_lab = self.prep_tran(image=edge_lab)['image']


        data = {'image': img, 'masks': [lab, edge_lab]}
        pics = self.transform(**data)
        img = pics['image'].float() / 255
        lab = pics['masks'][0].long()
        edge_lab = pics['masks'][1].long()

        # 处理edge_label


        # 处理sdf_label
        prob_labels = F.one_hot(lab).permute((2, 0, 1)).float()
        sdf_calculator = SDFCalculator(self.config, max_workers=4)  # 设置最大工作线程数
        sdf_label = sdf_calculator.compute_sdf(prob_labels)

        # show_tensor_img(img, 'img', 800)
        return img, {'label': lab, 'edge_label':edge_lab, 'sdf_label': sdf_label}, {'img_id': os.path.splitext(img_id)[0]}










