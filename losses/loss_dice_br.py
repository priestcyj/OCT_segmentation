import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CELoss, DiceLoss, Br_Loss


class Dice_Br_Loss(nn.Module):
    def __init__(self, config):
        super(Dice_Br_Loss, self).__init__()
        self.config = config
        self.dice_loss = DiceLoss(config)
        self.br_loss = Br_Loss(config)

    def forward(self, outputs, labels, **args):
        dice_loss = self.dice_loss(outputs['segment_output'], labels)
        br_loss = self.br_loss(outputs['boundary_output'], labels)
        dice_br_loss = dice_loss['total_loss'] + self.config['br_lamda'] * br_loss['total_loss']
        return {'total_loss': dice_br_loss, 'main_loss': dice_loss['total_loss'], 'aux_loss': br_loss['total_loss']}

