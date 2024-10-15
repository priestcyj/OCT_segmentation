import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CELoss, DiceLoss


class CE_Dice_Loss(nn.Module):
    def __init__(self, config):
        super(CE_Dice_Loss, self).__init__()
        self.config = config
        self.ce_loss = CELoss(config)
        self.dice_loss = DiceLoss(config)
        if 'ce_lamda' not in self.config:
            self.config['ce_lamda'] = 1

    def forward(self, outputs, labels, **args):
        if self.config['ce_lamda'] == 0:
            dice_loss = self.dice_loss(outputs, labels)
            return {'total_loss': dice_loss['total_loss'], 'main_loss': dice_loss['total_loss'], 'aux_loss': dice_loss['total_loss']}
        else:
            ce_loss = self.ce_loss(outputs, labels)
            dice_loss = self.dice_loss(outputs, labels)
            ce_dice_loss = self.config['ce_lamda'] * ce_loss['total_loss'] + self.config['dice_lamda'] * dice_loss['total_loss']
            return {'total_loss': ce_dice_loss, 'main_loss': ce_loss['total_loss'], 'aux_loss': dice_loss['total_loss']}

