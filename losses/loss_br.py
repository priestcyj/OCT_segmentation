import numpy as np
import platform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation as skimage_seg

from utils import show_outputs


class Br_Loss(nn.Module):
    def __init__(self, config, bi=False):
        super(Br_Loss, self).__init__()
        self.config = config

        self.loss = nn.MSELoss()
        self.tau = torch.tensor(config['br_tau'])
        self.frequency = 500
        self.num_calculations = 0


    def forward(self, outputs, labels):
        if isinstance(labels, dict):
            labels = labels['edge_label']
        if isinstance(outputs, dict):
            outputs = outputs['boundary_output']
        br_loss = self.regular_reg(outputs, labels).to(self.config['device'])
        return {'total_loss': br_loss}



    def sampling_softmax(self, outputs):
        # return F.softmax(outputs*self.tau.clamp(1,1000), dim=-2)
        eps = torch.rand_like(outputs, device=outputs.device)
        log_eps = torch.log(-torch.log(eps))
        # print('eps:', log_eps.min().item(), log_eps.max().item())
        gumbel_outputs = (outputs - log_eps) / self.tau.abs()  # self.tau.abs()
        gumbel_outputs = F.softmax(gumbel_outputs, dim=-2)
        # print('gumbel_outputs:', gumbel_outputs.shape)
        return gumbel_outputs / (1e-6 + gumbel_outputs.sum(dim=-2).unsqueeze(2))



    def regular_reg(self, outputs, labels, tau=100):
        B, C, H, W = outputs.shape
        prob_labels = F.one_hot(labels, C+1).permute((0, 3, 1, 2))[:, 1: , :, :].float()

        # edge_labels = torch.nonzero(prob_labels, as_tuple=False)
        edge_labels = torch.argmax((prob_labels == 1).int(), dim=2)

        # edge_labels, prob_labels = self.find_boundary(labels)
        edge_labels = edge_labels.to(labels.device) / H


        # soft_outputs = F.softmax(outputs, dim=-2)

        pseu_outputs = self.sampling_softmax(outputs)
        # pseu_outputs = F.softmax(outputs, dim=-2)

        idx_weight = torch.arange(0, H).reshape(1, 1, -1, 1).float().to(pseu_outputs.device)
        # idxp_weight = idx_weight
        idxp_weight = idx_weight + torch.rand_like(idx_weight).to(idx_weight.device) - 0.5
        edge_outputs = (pseu_outputs * idxp_weight).sum(dim=-2) / H

        #
        losEdge = self.loss(edge_outputs, edge_labels)
        # losProb = self.loss(soft_outputs, prob_labels)


        if (platform.system() == 'Windows') & (self.num_calculations % self.frequency == 0):
            soft_outputs = F.softmax(outputs.detach(), dim=-2)
            show_outputs(pseu_outputs.detach(), soft_outputs.detach(), prob_labels.detach())
        self.num_calculations += 1
        return losEdge


