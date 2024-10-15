import numpy as np
import platform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
from concurrent.futures import ThreadPoolExecutor

from model.backbone.projection import ProjectionHead
from utils import show_outputs


class Sdf_Loss(nn.Module):
    def __init__(self, config, is_weight=False):
        super(Sdf_Loss, self).__init__()
        self.config = config
        self.is_weight = is_weight
        if 'sdf_tau' in config:
            self.tau = config['sdf_tau']
        else:
            self.tau = 2

        self.num_classes = config['num_classes']
        self.loss = nn.MSELoss()

        self.num_calculations = 0
        self.frequency = 100
        self.max_workers = 16

        self.weight = None
        self.norm_weight = None

    def forward(self, outputs, label_dict):
        if isinstance(outputs, dict):
            outputs = outputs['sdf_output']
        sdf_label = label_dict['sdf_label']

        B,C,H,W = sdf_label.shape

        # -------------------------------process_outputs------------------------------------#
        tanh_pseu = F.tanh(outputs).to(outputs.device)


        #-------------------------------weight----------------------------------------------#
        sorted_sdf_output, _ = torch.sort(tanh_pseu.detach(), dim=1, descending=True)
        # sorted_sdf_output, _ = torch.sort(sdf_output.detach(), dim=1, descending=True)
        soft_sdf_output = F.softmax(sorted_sdf_output[:, 0:2, :, :] * self.tau, dim=1)

        def entropy(prob_dist):
            # 确保概率分布非零，避免log(0)的情况
            prob_dist = prob_dist + 1e-10
            return -torch.sum(prob_dist * torch.log(prob_dist), dim=1)

        entropy_sdf_output = entropy(soft_sdf_output)
        norm_entropy_sdf_output = (entropy_sdf_output / torch.log(torch.as_tensor(2.).to(sdf_label.device)))
        self.norm_weight = norm_entropy_sdf_output.unsqueeze(1)
        self.weight = norm_entropy_sdf_output.unsqueeze(1)

        if (platform.system() == 'Windows') and (self.num_calculations % self.frequency == 0):
            plt.imshow((norm_entropy_sdf_output[0]).cpu().numpy())
            plt.show()

        # -------------------------------loss-----------------------------------------------#
        if self.is_weight:
            loss = torch.square(tanh_pseu-sdf_label) * self.weight
            loss = torch.mean(loss)
        else:
            loss = self.loss(tanh_pseu, sdf_label)

        if (platform.system() == 'Windows') and (self.num_calculations % self.frequency == 0):
            norm_pseu = tanh_pseu.detach() / 2 + 0.5
            norm_sdf_label = sdf_label.detach() / 2 + 0.5

            plt.close('all')
            show_outputs(norm_pseu, norm_pseu, norm_sdf_label, idx1=1, idx2=2)
            # show_outputs(norm_pseu, norm_pseu, norm_sdf_label, idx1=3, idx2=4)
            # show_outputs(norm_pseu, norm_pseu, norm_sdf_label, idx1=5, idx2=6)
            # show_outputs(norm_pseu, norm_pseu, norm_sdf_label, idx1=7, idx2=8)


        self.num_calculations += 1

        return {'total_loss': loss}








