import os
import cv2
import platform
import numpy as np
import yaml
import torch
import argparse
import pandas as pd
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm
from PIL import Image
import albumentations as alb
import matplotlib.pyplot as plt

from dataset.aroi_dataset import AROI
from dataset.duke_dataset import DUKE
from dataset.hcms_dataset import HCMS
from dataset.heg_dataset import HEG
from utils import AverageMeter, str2bool
from model import *
from losses import *



def test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DUKE', help='model name')  # IBSR, MRBrainS13Data
    parser.add_argument('--checkpoint',
                        default=r'checkpoints/DUKE/unet_ce_loss/num_KFold_half_nb_filters_512')
    parser.add_argument('--save_format', default='png')
    return parser.parse_args()



def postprocess(output, config):
    argmax_output = torch.argmax(output, dim=1)[0]
    img = argmax_output.detach().cpu().numpy()
    post_tran = None
    if config['dataset'] == 'HCMS':
        post_tran = alb.Resize(p=1, height=128, width=1024, interpolation=cv2.INTER_NEAREST)
    elif config['dataset'] == 'HEG':
        post_tran = alb.Resize(p=1, height=256, width=512, interpolation=cv2.INTER_NEAREST)
    elif config['dataset'] == 'DUKE':
        post_tran = alb.Resize(p=1, height=256, width=512, interpolation=cv2.INTER_NEAREST)
    img_output = torch.tensor(post_tran(image=img)['image'])

    return img_output

def save_label_image_with_palette(image, output_path):
    """
    将标签图像保存为带有调色板的PNG文件。

    参数:
        label_image: 包含标签值的2D NumPy数组。
        output_path: 保存输出PNG文件的路径。
    """
    # 定义调色板 (R, G, B) 格式
    palette = [
        0, 0, 0,  # Black
        173, 216, 230,  # Light Blue
        0, 255, 255,  # Cyan
        0, 128, 0,  # Green
        255, 255, 0,  # Yellow
        255, 165, 0,  # Orange
        255, 0, 0,  # Red
        139, 0, 0,  # Dark Red
        0, 255, 0,  # Red
        0, 0, 255,  # Red
    ]

    # 创建PIL图像对象
    label_img_pil = Image.fromarray(image.astype(np.uint8), mode='P')

    # 应用调色板
    label_img_pil.putpalette(palette)

    # 保存图像
    label_img_pil.save(output_path)

def show_result(outputs, output_path):
    if outputs.dim() == 4:
        result_img = torch.argmax(outputs, dim=1).squeeze(0).detach().cpu().numpy()
    else:
        result_img = outputs.numpy()
    save_label_image_with_palette(result_img, output_path)


def save_heatmap(matrix, output_path):
    matrix = matrix.cpu().numpy()

    plt.imshow(matrix)
    plt.axis('off')  # 关闭坐标轴
    # plt.show()

    # 保存图像
    plt.imsave(output_path, matrix)

    # # 1. 将矩阵范围从 0 到 1 转换为 0 到 255
    # matrix_normalized = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX)
    # matrix_normalized = matrix_normalized.astype(np.uint8)
    #
    # # 2. 应用颜色映射
    # heatmap = cv2.applyColorMap(matrix_normalized, cv2.COLORMAP_HOT)
    #
    # # 3. 保存图像
    # cv2.imwrite(output_path, heatmap)


def evaluate(config, args, data_loader, model, criterion):
    avg_meters = {'total_loss': AverageMeter(),
                  'score': AverageMeter(),
                  'f1s': AverageMeter(),
                  'iou': AverageMeter()}
    with torch.no_grad():
        pbar = tqdm(total=len(data_loader))
        for images, labels, meta in data_loader:
            # save_resize_img
            img_name = meta['img_id'][0] + '.' + args.save_format
            images = images.to(config['device'], dtype=torch.float32)


            # 计算结果
            if isinstance(labels, dict):
                labels['label'] = labels['label'].to(config['device'], dtype=torch.long)
                labels['edge_label'] = labels['edge_label'].to(config['device'], dtype=torch.long)
                labels['sdf_label'] = labels['sdf_label'].to(config['device'], dtype=torch.float32)
            else:
                labels = labels.to(config['device'], dtype=torch.long)
            # compute output
            outputs = model(images)


            # save_model_output

            output_path_dir = os.path.join(r'D:\My_Project\yf_code\OCT_DATA',
                                           config['dataset'] + '_Result_Visualization')

            parent_name = os.path.basename(os.path.dirname(args.checkpoint))
            output_path = os.path.join(output_path_dir, parent_name, os.path.basename(args.checkpoint))
            if 'segment_output' in outputs:
                # output
                segment_output_path = os.path.join(output_path, 'segment')
                if not os.path.exists(segment_output_path):
                    os.makedirs(segment_output_path)
                segment_output_path = os.path.join(segment_output_path, img_name)
                show_result(outputs['segment_output'], segment_output_path)

                # postprocess_output
                resize_segment_output_path = os.path.join(output_path, 'resize_segment')
                if not os.path.exists(resize_segment_output_path):
                    os.makedirs(resize_segment_output_path)
                resize_segment_output_path = os.path.join(resize_segment_output_path, img_name)
                img_output = postprocess(outputs['segment_output'], config)
                show_result(img_output, resize_segment_output_path)

            if 'boundary_output' in outputs:
                boundary_output_path = os.path.join(output_path, 'boundary')
                if not os.path.exists(boundary_output_path):
                    os.makedirs(boundary_output_path)
                boundary_output_path = os.path.join(boundary_output_path, img_name)
                show_result(outputs['boundary_output'], boundary_output_path)

            if 'sdf' in config['loss']:
                losses = criterion(outputs, labels)
                weight = criterion.sdf_loss.norm_weight
                sdf_weight_path = os.path.join(output_path, 'sdf_weight')
                if not os.path.exists(sdf_weight_path):
                    os.makedirs(sdf_weight_path)
                sdf_weight_path = os.path.join(sdf_weight_path, img_name)
                save_heatmap(weight[0,0], sdf_weight_path)

            pbar.update(1)
        pbar.close()


###############################################main##############################################
def main():
    # checkpoint
    args = test_parse_args()
    config_path = os.path.join(args.checkpoint, 'config.yml')
    model_path = os.path.join(args.checkpoint, 'model.pth')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # 打印参数
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    if platform.system() == 'Windows':
        config['device'] = 'cuda:0'

    if config['dataset'] == 'HCMS':
        config['bds_num_classes'] = config['num_classes']
        config['sdf_num_classes'] = config['num_classes']



    # 创建模型并加载训练好的参数，如果是裁剪模型，直接加载模型及其参数
    backbones = {
        'unet': UNet,
        'nested_unet': NestedUNet,
        'att_unet': AttU_Net,
        'res_unet': ResUnet,
        'trans_unet': TransUNet,
        'tcct': stc_tt,

        'dh_unet': DH_UNet,
        'db_unet_connect_false': DB_UNet_ConnectFalse,
        'db_unet_connect_true': DB_UNet_ConnectTrue,
        'db_unet_channel_attention':
            DoubleBranchChannelAttentionUNet,
        'db_unet_cross_attention':
            DoubleBranchCrossAttentionUNet,

        'dh_res_unet': DH_Res_UNet,
        'db_res_unet_connect_false': DB_Res_UNet_ConnectFalse,
        'db_res_unet_connect_true': DB_Res_UNet_ConnectTrue,

        'res_db_unet_cross_attention':
            ResDoubleBranchCrossAttentionUNet,
        'res_db_unet_cross_attention_with_channel_attention':
            ResDoubleBranchCrossAttentionUNetWithChannelAttention
    }
    model = backbones[config['backbone']](config)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model = model.to(config['device'])
    model.eval()

    # define losses function (criterion)
    loss = {
        'ce_loss': CELoss,
        'dice_loss': DiceLoss,
        'ce_dice_loss': CE_Dice_Loss,

        'dice_br_loss': Dice_Br_Loss,

        'dice_sdf_loss': Dice_Sdf_Loss,
        'dice_weight_sdf_loss': Dice_Weight_Sdf_Loss,

        'ce_dice_sdf_loss': CE_Dice_Sdf_Loss,
        'ce_dice_weight_sdf_loss': CE_Dice_Weight_Sdf_Loss
    }
    criterion = loss[config['loss']](config)

    criterion = criterion.to(config['device'])
    cudnn.benchmark = True

    # Data loading code
    txt = os.path.join('input_division', config['dataset'], config['num_KFold'])
    train_img_ids = []
    val_img_ids = []
    test_img_ids = []
    with open(os.path.join(txt, 'Train_images.txt'), 'r') as file:
        for line in file:
            train_img_ids.append(line.strip('\n').split()[0])
    with open(os.path.join(txt, 'Val_images.txt'), 'r') as file:
        for line in file:
            val_img_ids.append(line.strip('\n').split()[0])
    with open(os.path.join(txt, 'Test_images.txt'), 'r') as file:
        for line in file:
            test_img_ids.append(line.strip('\n').split()[0])

    lens = {
        'train': len(train_img_ids),
        'val': len(val_img_ids),
        'test': len(test_img_ids)
    }
    with open(os.path.join(txt, 'Test_images.txt'), 'r') as file:
        for line in file:
            test_img_ids.append(line.strip('\n').split()[0])

    Dataset = {
        'HCMS': HCMS,
        'AROI': AROI,
        'DUKE': DUKE,
        'HEG': HEG
    }

    test_dataset = Dataset[config['dataset']](
        config=config,
        img_ids=test_img_ids,
        num_classes=config['num_classes'],
        lens=lens,
        mode='test'
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)


    # evaluate
    evaluate(config, args, test_loader, model, criterion)

    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()


