import os
import platform

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm

from dataset.aroi_dataset import AROI
from dataset.duke_dataset import DUKE
from dataset.hcms_dataset import HCMS
from dataset.heg_dataset import HEG
from utils import AverageMeter, str2bool, count_params, show_tensor_img, show_tensor_img_click, show_outputs
from model import *
from losses import *

def test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DUKE', help='model name')  # IBSR, MRBrainS13Data
    parser.add_argument('--checkpoint', 
                        default=r'checkpoints/HCMS/best_net/res_db_unet_cross_attention_with_channel_attention_dice_weight_sdf_loss')
    parser.add_argument('--is_pruned', default=True, type=str2bool)
    return parser.parse_args()

def contrast_sdf_seg(seg_outputs, sdf_output, label, meta):
    soft_seg_outputs = F.softmax(seg_outputs, dim=1)
    sorted_sdf_output, _ = torch.sort(sdf_output, dim=1, descending=True)

    soft_sdf_output = F.softmax(sorted_sdf_output[:, 0:2, :, :], dim=1)

    def entropy(prob_dist):
        # 确保概率分布非零，避免log(0)的情况
        prob_dist = prob_dist + 1e-10
        return -torch.sum(prob_dist * torch.log(prob_dist), dim=1)

    entropy_sdf_output = entropy(soft_sdf_output)
    norm_entropy_sdf_output = (entropy_sdf_output / torch.log(torch.tensor(2.))).cpu().numpy()
    num_classes = soft_seg_outputs.shape[1]
    argmax_seg_outputs = torch.argmax(soft_seg_outputs, dim=1)
    one_hot_seg_outputs = F.one_hot(argmax_seg_outputs, num_classes=num_classes).contiguous().permute(0, 3, 1, 2).cpu() * 255

    for i in range(one_hot_seg_outputs.shape[1]):
        print(meta['img_id'][0]+'_class_'+str(i))
        # show_tensor_img_click(one_hot_seg_outputs[0,i], soft_seg_outputs, sdf_output, label,'sdf', width=800)

    plt.imshow(norm_entropy_sdf_output[0])
    plt.show()
    # sdf_output = sdf_output / 2 + 0.5
    # one_hot_sdf_result = (sdf_output >= 0.5).int().cpu() * 255
    # img = torch.cat([one_hot_seg_outputs, one_hot_sdf_result], dim=2)
    # for i in range(img.shape[1]):
    #     show_tensor_img(img[0,i], 'sdf', width=800)



def evaluate(config, data_loader, model, criterion, class_name):
    avg_meters = {'total_loss': AverageMeter(),
                  'f1s': AverageMeter(),
                  'iou': AverageMeter()}
    for i in range(len(class_name)):
        avg_meters[class_name[i]] = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(total=len(data_loader))
        for images, labels, meta in data_loader:
            images = images.to(config['device'], dtype=torch.float32)
            if isinstance(labels, dict):
                labels['label'] = labels['label'].to(config['device'], dtype=torch.long)
                labels['edge_label'] = labels['edge_label'].to(config['device'], dtype=torch.long)
                labels['sdf_label'] = labels['sdf_label'].to(config['device'], dtype=torch.float32)
            else:
                labels = labels.to(config['device'], dtype=torch.long)
            # compute output
            outputs = model(images)

            # losses
            losses = criterion(outputs, labels)
            # if 'sdf' in config['loss']:
            #     if 'hc03_spectralis_macula_v1_s1_R_42' in meta['img_id'][0]:
            #         weight = criterion.weight
            #         plt.imshow(weight[0, 0].cpu().numpy())
            #         plt.show()
            # IOU & draw
            # if isinstance(outputs, dict):
            #     seg_outputs = outputs['segment_output']
            #     sdf_output = outputs['sdf_output']
            #     br_output = outputs['boundary_output']
            #     contrast_sdf_seg(seg_outputs, sdf_output, labels['label'], meta)



            # outputs_detach = outputs['segment_output']
            outputs_detach = outputs['segment_output']
            f1s = MDiceLoss.scorem(outputs_detach, labels, start_idx=1).cpu().item()
            iou = MIouLoss.scorem(outputs_detach, labels, start_idx=1).cpu().item()
            score = np.array(MDiceLoss.scores(outputs_detach, labels)).reshape(-1).astype(np.float32)



            avg_meters['total_loss'].update(losses['total_loss'].item(), images.size(0))
            for j in range(len(class_name)):
                avg_meters[class_name[j]].update(score[j], images.size(0))
            avg_meters['f1s'].update(f1s, images.size(0))
            avg_meters['iou'].update(iou, images.size(0))


            postfix = OrderedDict([
                ('total_loss', avg_meters['total_loss'].avg),
                ('f1s', avg_meters['f1s'].avg),
                ('iou', avg_meters['iou'].avg),
            ])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return avg_meters

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


    # 模型结果存储路径
    results_dir = os.path.join(args.checkpoint.replace('checkpoints', 'results'))
    os.makedirs(results_dir, exist_ok=True)

    if config['dataset'] == 'HCMS':
        config['bds_num_classes'] = config['num_classes']
        config['sdf_num_classes'] = config['num_classes']


    # 创建模型并加载训练好的参数，如果是裁剪模型，直接加载模型及其参数
    if not args.is_pruned:
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
    else:
        model_basename = os.path.basename(model_path)
        model = torch.load(model_path.replace(model_basename, 'model_tuned_iter_10'), map_location=lambda storage, loc: storage)
    model = model.to(config['device'])
    model.eval()

    print(f'Number of parameters: {count_params(model)}')

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

    class_name = []
    if config['dataset'] == 'HCMS':
        class_name = ['BG', 'RNFL', 'GCIPL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE']
    elif config['dataset'] == 'DUKE':
        class_name = ['BG', 'RNFL', 'GCIPL', 'INL', 'OPL', 'ONL', 'IS', 'OS-RPE']
    elif config['dataset'] == 'AROI':
        class_name = ['Above ILM', 'ILM-IPL/INL', 'IPL/INL-RPE', 'RPE-BM', 'Under BM', 'PED', 'SRF', 'IRF']
    elif config['dataset'] == 'HEG':
        class_name = ['BG', 'OS-PRE', 'IS', 'ONL', 'OPL', 'INL', 'GCIPL', 'RNFL']

    # evaluate
    test_log = evaluate(config, test_loader, model, criterion, class_name)

    torch.cuda.empty_cache()
    # 每次实验在训练集、验证集、测试集上的结果
    result = {'total_loss': [test_log['total_loss'].avg]}

    for i in range(len(class_name)):
        result[class_name[i]] = [test_log[class_name[i]].avg]
    result['f1s'] = [test_log['f1s'].avg]
    result['iou'] = [test_log['iou'].avg]

    df_result = pd.DataFrame(result)
    df_result.to_excel(os.path.join(results_dir, 'result.xlsx'), index=False)


if __name__ == '__main__':
    main()


