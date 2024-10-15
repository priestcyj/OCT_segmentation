import time

import cv2
import platform
if platform.system() == 'Windows':
    import pyautogui
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def total_num_filters(model):
    filters = 0
    for _, block in model._modules.items():
        if not isinstance(block, torch.nn.Upsample) and not isinstance(block, torch.nn.MaxPool2d):
            for _, module in block._modules.items():
                if isinstance(module, torch.nn.Conv2d):
                    filters = filters + module.out_channels
    return filters

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def show_histogram(param_data, name):
    plt.hist(param_data, bins=50)
    plt.title(name)
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    plt.show()


def show_outputs(a, b, c, idx1=1, idx2=-1):

    plt.subplot(231), plt.imshow(a[0, idx1].data.cpu().numpy())
    plt.subplot(232), plt.imshow(b[0,idx1].data.cpu().numpy())
    plt.subplot(233), plt.imshow(c[0,idx1].data.cpu().numpy())

    plt.subplot(234), plt.imshow(a[0, idx2].data.cpu().numpy())
    plt.subplot(235), plt.imshow(b[0,idx2].data.cpu().numpy())
    plt.subplot(236), plt.imshow(c[0,idx2].data.cpu().numpy())

    plt.show()

def display_colored_image(label, window_name, width=800):
    img = np.zeros((label.shape[0], label.shape[1], 3)).astype('uint8')
    colors = [
        (0, 0, 0),  # Black
        (173, 216, 230),
        (0, 255, 255),
        (0, 128, 0),
        (255, 255, 0),
        (255, 165, 0),
        (255, 0, 0),
        (139, 0, 0),
        (0, 0, 255)  # Red
    ]

    for c in range(len(np.unique(label))):
        img[label == c] = colors[c]

    height = int(width * img.shape[0]/img.shape[1])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 计算窗口位置，使其位于屏幕中央
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # 移动窗口到中心位置
    cv2.moveWindow(window_name, x, y)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    # cv2.destroyAllWindows()

def show_img(img, window_name, width=800):
    height = int(width * img.shape[0]/img.shape[1])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 计算窗口位置，使其位于屏幕中央
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # 移动窗口到中心位置
    cv2.moveWindow(window_name, x, y)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    # cv2.destroyAllWindows()



def show_tensor_img(data_tensor, window_name, width=800):
    img = np.array(data_tensor).astype('uint8').squeeze()
    height = int(width * img.shape[0]/img.shape[1])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 计算窗口位置，使其位于屏幕中央
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # 移动窗口到中心位置
    cv2.moveWindow(window_name, x, y)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show_tensor_img_click(one_hot_data_tensor, soft_seg_outputs, sdf_outputs, label, window_name, width=800):
    img = np.array(one_hot_data_tensor).astype('uint8').squeeze()
    height = int(width * img.shape[0]/img.shape[1])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 计算窗口位置，使其位于屏幕中央
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    # 移动窗口到中心位置
    cv2.moveWindow(window_name, x, y)
    coordinates = []
    cv2.setMouseCallback(window_name, mouse_callback, coordinates)

    while True:
        cv2.imshow(window_name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    if len(coordinates) != 0:
        print(soft_seg_outputs[0, :, coordinates[0][1], coordinates[0][0]])
        print(sdf_outputs[0, :, coordinates[0][1], coordinates[0][0]])
        print(label[0, coordinates[0][1], coordinates[0][0]])





def mouse_callback(event, click_x, click_y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 如果是左键点击事件
        print(f"Mouse clicked at ({click_x}, {click_y})")
        param.append((click_x, click_y))












