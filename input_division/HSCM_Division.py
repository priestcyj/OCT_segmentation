import os
import re
from glob import glob
from sklearn.model_selection import train_test_split, KFold
import yaml
import json
import cv2
import numpy as np



dir_path = r'D:\My_Project\yf_code\OCT_DATA\HCMS_Processed_by_Dr_Fan_Yang'
img_dir_path = os.path.join(dir_path, 'image')

img_list = os.listdir(img_dir_path)

name_list = [os.path.basename(img) for img in img_list]

hc_string = 'hc'
ms_string = 'ms'
hc_pattern = re.compile(re.escape(hc_string) + r'(\d+)')
ms_pattern = re.compile(re.escape(ms_string) + r'(\d+)')

kf = KFold(n_splits=5, shuffle=True, random_state=1)
pth = r'HCMS'

train_val_list = []
train_list = []
val_list = []
test_list = []
for name in name_list:
    if hc_pattern.search(name) is not None:
        hc_id = int(hc_pattern.search(name).group(1))
        if hc_id <= 8:
            test_list.append(name)
        else:
            train_val_list.append(name)
    elif ms_pattern.search(name) is not None:
        ms_id = int(ms_pattern.search(name).group(1))
        if ms_id <= 12:
            test_list.append(name)
        else:
            train_val_list.append(name)




i = 1
for train_index, val_index in kf.split(train_val_list):
    train_list, val_list = [train_val_list[i] for i in train_index], [train_val_list[i] for i in val_index]

    os.makedirs(os.path.join(pth, 'kf_' + str(i)), exist_ok=True)
    #创建一个指定路径
    train_file = open(os.path.join(pth, 'kf_' + str(i), 'Train_images.txt'), 'w')
    val_file = open(os.path.join(pth, 'kf_' + str(i), 'Val_images.txt'), 'w')
    test_file = open(os.path.join(pth, 'kf_' + str(i), 'Test_images.txt'), 'w')
    #t是一个整形，不能直接与字符串进行拼接。所以，需要将t转换成字符串类型后再进行拼接。
    for t in train_list:
        train_file.writelines(str(t) + '\n')
    for t in val_list:
        val_file.writelines(str(t) + '\n')
    for t in test_list:
        test_file.writelines(str(t) + '\n')
    i += 1


