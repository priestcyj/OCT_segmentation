import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import CELoss, DiceLoss, Br_Loss, Sdf_Loss


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

class Dice_Sdf_Loss(nn.Module):
    def __init__(self, config, is_weight=False):
        super(Dice_Sdf_Loss, self).__init__()
        self.config = config
        self.dice_loss = DiceLoss(config)
        self.sdf_loss = Sdf_Loss(config, is_weight)

    def forward(self, outputs, labels, **args):
        sdf_loss = self.sdf_loss(outputs, labels)
        dice_loss = self.dice_loss(outputs['segment_output'], labels)

        dice_br_loss = dice_loss['total_loss'] + self.config['sdf_lamda'] * sdf_loss['total_loss']

        return {'total_loss': dice_br_loss, 'main_loss': dice_loss['total_loss'], 'aux_loss': sdf_loss['total_loss']}


class CE_Dice_Sdf_Loss(nn.Module):
    def __init__(self, config, is_weight=False):
        super(CE_Dice_Sdf_Loss, self).__init__()
        self.config = config
        self.ce_loss = CELoss(config)
        self.dice_loss = DiceLoss(config)
        self.sdf_loss = Sdf_Loss(config, is_weight)

    def forward(self, outputs, labels, **args):
        sdf_loss = self.sdf_loss(outputs['sdf_output'], labels)
        self.ce_loss.spatial_weight = self.sdf_loss.weight

        ce_loss = self.ce_loss(outputs['segment_output'], labels)
        dice_loss = self.dice_loss(outputs['segment_output'], labels)


        ce_sdf_loss = (self.config['ce_lamda'] * ce_loss['total_loss']
                       + self.config['dice_lamda'] * dice_loss['total_loss']
                       + self.config['sdf_lamda'] * sdf_loss['total_loss'])
        return {'total_loss': ce_sdf_loss, 'main_loss': ce_loss['total_loss'], 'aux_loss': sdf_loss['total_loss']}



class Dice_Weight_Sdf_Loss(Dice_Sdf_Loss):
    def __init__(self, config, **kwargs):
        super().__init__(config, is_weight=True, **kwargs)

class CE_Dice_Weight_Sdf_Loss(CE_Dice_Sdf_Loss):
    def __init__(self, config, **kwargs):
        super().__init__(config, is_weight=True, **kwargs)