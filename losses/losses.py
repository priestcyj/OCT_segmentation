import cv2
import numpy
import os

import numpy as np
from numpy import array

import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import segmentation as skimage_seg
from scipy.ndimage import filters
from scipy.ndimage import distance_transform_edt as distance



def ce_seg_bou_loss(output, label_segment, lamda, meta = None):
    seg = ce_loss(output[0], label_segment)

    label_onehot = F.one_hot(label_segment).permute(0, 3, 1, 2)   # (batch_size,4,256,128)
    # label_boundary = numpy.zeros(label_segment.shape)
    label_onehot_boundary = numpy.zeros(label_onehot.shape)
    for i in range(label_segment.shape[0]):  # batch_size
        # 全部边界
        # label_boundary[i, :, :] = skimage_seg.find_boundaries(label_segment[i, :, :].cpu().numpy(), mode='thick').astype('int32')
        # cv2.imwrite(os.path.join('inputs', 'images', 'boundary', 'all', meta['img_id'][i]+'.png'), (label_boundary[i, :, :]*255).astype('uint8'))
        for j in range(label_onehot.shape[1]):  # class
            # 每一个目标的边界
            label_onehot_boundary[i, j, :, :] = skimage_seg.find_boundaries(label_onehot[i, j, :, :].cpu().numpy(), mode='inner').astype('int32')
            # cv2.imwrite(os.path.join('inputs', 'images', 'boundary', str(j), meta['img_id'][i] + '.png'), (label_onehot_boundary[i, j, :, :] * 255).astype('uint8'))
    # label_boundary = torch.from_numpy(label_boundary).to('cuda', dtype=torch.long)
    label_onehot_boundary = torch.from_numpy(label_onehot_boundary).to('cuda', dtype=torch.float)
    bou = F.binary_cross_entropy_with_logits(output[1], label_onehot_boundary)

    return seg + lamda * bou, seg, bou

def ce_seg_sdf_bou_loss(output, label_segment, lamda=1.5, meta=None, image=None):
    seg = ce_loss(output[0], label_segment)
    bou = sdf_loss(output[1], label_segment, meta=meta)
    return seg + lamda * bou * 10, seg, bou

# 定义水平距离函数回归函数
def sdf_loss(output, label, meta):
    label_onehot = F.one_hot(label).permute(0, 3, 1, 2)   # (batch_size,4,256,128)
    label_onehot = label_onehot.cpu().numpy()
    label_sdf = np.zeros(label_onehot.shape).astype('float32')
    output_tanh = torch.tanh(output)
    w = torch.tensor([1, 1, 1, 1]).to(device='cuda')
    for i in range(4):
        label_sd = label_onehot[:, i, :, :]
        label_sd_shape = label_sd.shape
        hm = compute_sdf(label_sd, label_sd_shape)
        label_sdf[:, i, :, :] = hm

        # # 输出sdf标签
        # os.makedirs(os.path.join('inputs', 'images', 'sdf', str(i)), exist_ok=True)
        # for j in range(len(meta['img_id'])):
        #     heatmap = (-hm[j] + 1) / 2
        #     heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #     cv2.imwrite(os.path.join('inputs', 'images', 'sdf', str(i), meta['img_id'][j]+'.jpg'), heatmap)


    label_sdf = torch.from_numpy(label_sdf).cuda()
    loss = (label_sdf - output_tanh)**2
    loss = torch.mean(loss, dim=(2, 3))
    loss = torch.mean(loss, dim=0)
    loss = torch.sum(loss * w)/4
    return loss


# 计算符号距离函数
def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-a2,a2]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        threshold = 1000
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            posdis = np.clip(posdis, a_min=None, a_max=threshold)
            negdis = np.clip(negdis, a_min=None, a_max=threshold)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf

    return normalized_sdf




# gauss窗拓宽边界
def ce_seg_bou_gaussian_loss(output, label_segment, lamda, meta=None, image=None):
    seg = ce_loss(output[0], label_segment)

    label_onehot = F.one_hot(label_segment).permute(0, 3, 1, 2)  # (batch_size,4,256,128)
    label_boundary = numpy.zeros(label_segment.shape)
    label_onehot_boundary = numpy.zeros(label_onehot.shape)
    sigma = 2
    for i in range(label_segment.shape[0]):  # batch_size
        # 全部边界
        # label_boundary[i, :, :] = skimage_seg.find_boundaries(label_segment[i, :, :].cpu().numpy(), mode='thick').astype('int32')
        # label_boundary[i, :, :] = filters.gaussian_filter(sigma * label_boundary[i, :, :], 2)
        # cv2.imwrite(os.path.join('inputs', 'images', 'gauss_boundary', 'all', meta['img_id'][i] + '.png'), (label_boundary[i, :, :] * 255).astype('uint8'))
        for j in range(label_onehot.shape[1]):
            # 每一个目标的边界
            label_onehot_boundary[i, j, :, :] = skimage_seg.find_boundaries(label_onehot[i, j, :, :].cpu().numpy(), mode='inner').astype('int32')
            label_onehot_boundary[i, j, :, :] = filters.gaussian_filter(sigma * label_onehot_boundary[i, j, :, :], 2)
            # cv2.imwrite(os.path.join('inputs', 'images', 'gauss_boundary', str(j), meta['img_id'][i] + '.png'), (label_onehot_boundary[i, j, :, :] * 255).astype('uint8'))
    # label_boundary = torch.from_numpy(label_boundary).to('cuda', dtype=torch.long)
    label_onehot_boundary = torch.from_numpy(label_onehot_boundary).to('cuda', dtype=torch.float)
    bou = F.binary_cross_entropy_with_logits(output[1], label_onehot_boundary)

    return seg + lamda * bou, seg, bou



if __name__ == '__main__':
    device = torch.device('cuda')

