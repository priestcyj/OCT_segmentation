import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class CELoss(nn.Module):
    def __init__(self, config, bi=False):
        super(CELoss, self).__init__()
        self.config = config
        self.class_weight = torch.tensor(self.config['ce_weight']).to(config['device'])
        self.spatial_weight = None

    def forward(self, outputs, labels):
        if isinstance(labels, dict):
            labels = labels['label']
        if isinstance(outputs, dict):
            outputs = outputs['segment_output']


        # 获取batch size和类别数
        B, C, H, W = outputs.shape

        # 计算标准交叉熵损失
        loss = F.cross_entropy(outputs, labels, weight= self.class_weight, reduction='none').unsqueeze(1)

        # 将空间权重应用到损失上
        if self.spatial_weight is not None:
            loss = loss * self.spatial_weight

        # 计算平均损失
        loss = torch.mean(loss)

        return {'total_loss': loss}






