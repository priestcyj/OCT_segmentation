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
from skimage import segmentation as skimage_seg


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


def crop_rows_to_foreground(img, label_img, edge_lab, target_height=512):
    # 1. 创建掩码：标记前景区域（非背景）
    foreground_mask = np.logical_and(label_img != 0, label_img != 4)

    # 2. 找到前景区域的行边界
    rows = np.any(foreground_mask, axis=1)

    # 获取前景的最小边界
    r_min, r_max = np.where(rows)[0][[0, -1]]

    # 计算裁剪区域的起始和结束行
    if r_max - r_min + 1 <= target_height:
        # 如果前景高度小于或等于目标高度，中心裁剪
        r_start = max(r_min - (target_height - (r_max - r_min + 1)) // 2, 0)
        r_end = r_start + target_height
        if r_end > label_img.shape[0]:
            r_end = label_img.shape[0]
            r_start = r_end - target_height
    else:
        # 如果前景高度大于目标高度，按前景中心裁剪
        r_start = max(r_min, 0)
        r_end = r_start + target_height

    # 3. 裁剪图像
    cropped_img = img[r_start:r_end, :]
    cropped_label = label_img[r_start:r_end, :]
    cropped_edge_lab = edge_lab[r_start:r_end, :]

    return cropped_img, cropped_label, cropped_edge_lab

def find_boundary(label):
    labels = F.one_hot(label).numpy()
    C = 5
    boundary = np.zeros_like(labels[:,:,:C])
    height, width = labels.shape[0], labels.shape[1]
    for c in range(1, C):
        for col in range(width):
            for row in range(height):
                if labels[row, col, c] == 1:
                    boundary[row, col, c] = 1
                    break

    prob_labels = boundary
    edge_labels = np.argmax(prob_labels == 1, axis=-1)
    return edge_labels

class AROI(torch.utils.data.Dataset):
    def __init__(self, config, img_ids, num_classes, lens, mode=None):
        self.config = config
        self.patient_ids = img_ids
        if platform.system() == 'Windows':
            self.data_dir = r'D:\My_Project\yf_code\OCT_DATA\AROI'
        else:
            self.data_dir = r'/data/yf/OCT_DATA/AROI'
        self.num_classes = num_classes
        self.mode = mode

        self.patient_dir = os.path.join(self.data_dir, '24_patient')
        self.patient_dir_list = [os.path.join(self.patient_dir, patient) for patient in self.patient_ids]

        self.img_dir_list = [os.path.join(patient_dir, 'raw', 'labeled') for patient_dir in self.patient_dir_list]
        self.label_dir_list = [os.path.join(patient_dir, 'mask', 'number') for patient_dir in self.patient_dir_list]
        self.edge_dir_list = [os.path.join(patient_dir, 'edge') for patient_dir in self.patient_dir_list]

        def get_all_files_from_paths(paths):
            all_files = []
            for directory in paths:
                if os.path.exists(directory) and os.path.isdir(directory):
                    files = [os.path.join(directory, f)
                             for f in os.listdir(directory)
                             if os.path.isfile(os.path.join(directory, f))]
                    all_files.extend(files)
            return all_files

        self.img_list = get_all_files_from_paths(self.img_dir_list)
        self.lab_list = get_all_files_from_paths(self.label_dir_list)
        self.edge_list = get_all_files_from_paths(self.edge_dir_list)


        self.prep_tran = alb.Resize(p=1, height=512, width=256, interpolation=cv2.INTER_NEAREST)
        self.transform = make_tran(512, 256, mode)

        self.lens = len(self.img_list)

        if mode == 'train':
            self.exeNums = max(1, 735 // self.lens)
        elif mode == 'val':
            self.exeNums = 1
        elif mode == 'test':
            self.exeNums = 1



    def __len__(self):
        if self.mode == 'test':
            return self.lens
        elif self.mode == 'val':
            return self.lens
        return self.lens * self.exeNums

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        lab_path = self.lab_list[idx]
        edge_path = self.edge_list[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
        edge_lab = np.array(Image.open(edge_path).convert('P'))

        # img, lab, edge_lab = crop_rows_to_foreground(img, lab, edge_lab, target_height=512)

        # 数据增强
        img = self.prep_tran(image=img)['image']
        lab = self.prep_tran(image=lab)['image']

        edge_lab = self.prep_tran(image=edge_lab)['image']


        data = {'image': img, 'masks': [lab, edge_lab]}
        pics = self.transform(**data)
        img = pics['image'].float() / 255
        lab = pics['masks'][0].long()
        edge_lab = pics['masks'][1].long()


        edge_lab = find_boundary(lab)
        edge_lab = torch.as_tensor(edge_lab)

        # 处理sdf_label
        # print(os.path.splitext(os.path.basename(img_path))[0])
        prob_labels = F.one_hot(lab, self.num_classes).permute((2, 0, 1)).float()
        prob_labels = prob_labels[:-3,:,:]
        sdf_calculator = SDFCalculator(self.config, max_workers=4)  # 设置最大工作线程数
        sdf_label = sdf_calculator.compute_sdf(prob_labels)

        # show_tensor_img(img, 'img', 400)
        # show_tensor_img(lab*30, 'lab', 400)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        return img, {'label': lab, 'edge_label':edge_lab, 'sdf_label': sdf_label}, {'img_id': img_id}










