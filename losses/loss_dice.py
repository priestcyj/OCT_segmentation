import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, config, bi=False):
        super(DiceLoss, self).__init__()
        self.config = config
        self.func = self.dice2

    def forward(self, outputs, labels, **args):
        if isinstance(labels, dict):
            labels = labels['label']
        if isinstance(outputs, dict):
            outputs = outputs['segment_output']
        outputs = F.softmax(outputs, dim=1)
        dice_loss = 1 - self.func(outputs, labels)
        return {'total_loss': dice_loss}

    @staticmethod
    def dice2(outputs, labels, smooth=1):
        labels = F.one_hot(labels).permute((0, 3, 1, 2))
        outputs, labels = outputs.reshape(-1), labels.reshape(-1)
        inter = (outputs * labels).sum()
        union = (outputs ** 2 + labels ** 2).sum()
        return (smooth + 2 * inter) / (smooth + union)

    @staticmethod
    def dice(outputs, labels, smooth=1):
        labels = F.one_hot(labels).permute((0, 3, 1, 2))
        outputs, labels = outputs.reshape(-1), labels.reshape(-1)
        inter = (outputs * labels).sum()
        union = (outputs + labels).sum()
        return (smooth + 2 * inter) / (smooth + union)

    @staticmethod
    def dicem(outputs, labels, smooth=1e-6):
        labels = F.one_hot(labels).permute((0, 3, 1, 2))
        # outputs = F.softmax(outputs, dim=1).round().long()
        score = sum([DiceLoss.dice(outputs[:, i:i + 1], labels[:, i:i + 1]) for i in range(outputs.shape[1])])
        return score / outputs.shape[1]


class IouLoss(nn.Module):
    def __init__(self, config, bi=False):
        super(IouLoss, self).__init__()
        self.config = config
        self.bi = bi

    def forward(self, outputs, labels, **args):
        iou_loss = 1 - self.iou(outputs, labels)
        return {'total_loss': iou_loss}

    @staticmethod
    def iou(outputs, labels, smooth=1e-12):
        labels = F.one_hot(labels).permute((0, 3, 1, 2))
        outputs, labels = outputs.reshape(-1), labels.reshape(-1)
        inter = (outputs * labels).sum()
        union = (outputs + labels).sum() - inter
        return (inter + smooth) / (union + smooth)

    @staticmethod
    def miou(outputs, labels, ignore_index=0):
        labels = F.one_hot(labels).permute((0, 3, 1, 2))
        assert 0 <= outputs.max().item() <= 1, 'prediction is not binary!'
        assert 0 <= labels.shape[1] <= 9, 'labels lesion class is wrong!'
        assert 0 <= outputs.shape[1] <= 9, 'outputs lesion class is wrong!'
        mious = []
        for i in range(labels.shape[1]):
            if i == ignore_index:
                continue
            mious.append(IouLoss.iou(outputs[:, i], labels[:, i]).item())
        losStr = ','.join(['{:.4f}'.format(it) for it in mious])
        return sum(mious) / len(mious), losStr


