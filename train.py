import os
import time
import numpy as np
import yaml
import torch
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.hcms_dataset import HCMS
from dataset.aroi_dataset import AROI
from dataset.duke_dataset import DUKE
from dataset.heg_dataset import HEG
from utils import AverageMeter, count_params
from Config.config_hcms import parse_args as hcms_args
from Config.config_aroi import parse_args as aroi_args
from Config.config_duke import parse_args as duke_args
from Config.config_heg import parse_args as heg_args

from model import *
from losses import *

dataset = 'duke'
cudnn.benchmark = True

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'train_loss': AverageMeter(),
                  'train_main_loss': AverageMeter(),
                  'train_aux_loss': AverageMeter(),
                  'train_f1s': AverageMeter(),
                  'train_iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for images, labels, _ in train_loader:
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
        total_loss = losses['total_loss']

        # iou
        if isinstance(outputs, dict):
            seg_outputs = outputs['segment_output']
        outputs_detach = seg_outputs.detach()
        f1s = MDiceLoss.scorem(outputs_detach, labels, start_idx=1).cpu().item()
        iou = MIouLoss.scorem(outputs_detach, labels, start_idx=1).cpu().item()

        # compute gradient and do optimizing step
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()
        optimizer.zero_grad()



        # 记录在summary_write
        if 'main_loss' not in losses.keys():
            losses['main_loss'] = losses['total_loss']
            losses['aux_loss'] = losses['total_loss']

        avg_meters['train_loss'].update(losses['total_loss'].item(), images.size(0))
        avg_meters['train_main_loss'].update(losses['main_loss'].item(), images.size(0))
        avg_meters['train_aux_loss'].update(losses['aux_loss'].item(), images.size(0))
        avg_meters['train_f1s'].update(f1s, images.size(0))
        avg_meters['train_iou'].update(iou, images.size(0))

        postfix = OrderedDict([
            ('train_loss', avg_meters['train_loss'].avg),
            ('train_main_loss', avg_meters['train_main_loss'].avg),
            ('train_aux_loss', avg_meters['train_aux_loss'].avg),
            ('train_f1s', avg_meters['train_f1s'].avg),
            ('train_iou', avg_meters['train_iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return postfix

def validate(config, val_loader, model, criterion):
    avg_meters = {'val_loss': AverageMeter(),
                  'val_main_loss': AverageMeter(),
                  'val_aux_loss': AverageMeter(),
                  'val_f1s': AverageMeter(),
                  'val_iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for images, labels, _ in val_loader:
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

            # iou
            if isinstance(outputs, dict):
                seg_outputs = outputs['segment_output']
            outputs_detach = seg_outputs.detach()
            f1s = MDiceLoss.scorem(outputs_detach, labels, start_idx=1).cpu().item()
            iou = MIouLoss.scorem(outputs_detach, labels, start_idx=1).cpu().item()

            if 'main_loss' not in losses.keys():
                losses['main_loss'] = losses['total_loss']
                losses['aux_loss'] = losses['total_loss']

            avg_meters['val_loss'].update(losses['total_loss'].item(), images.size(0))
            avg_meters['val_main_loss'].update(losses['main_loss'].item(), images.size(0))
            avg_meters['val_aux_loss'].update(losses['aux_loss'].item(), images.size(0))
            avg_meters['val_f1s'].update(f1s, images.size(0))
            avg_meters['val_iou'].update(iou, images.size(0))

            postfix = OrderedDict([
                ('val_loss', avg_meters['val_loss'].avg),
                ('val_main_loss', avg_meters['val_main_loss'].avg),
                ('val_aux_loss', avg_meters['val_aux_loss'].avg),
                ('val_f1s', avg_meters['val_f1s'].avg),
                ('val_iou', avg_meters['val_iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return postfix

# ------------------------------------------main()------------------------------------------------#
def main():
    # 相关参数输入
    if dataset == 'hcms':
        config = vars(hcms_args())
    elif dataset == 'aroi':
        config = vars(aroi_args())
    elif dataset == 'duke':
        config = vars(duke_args())
    elif dataset == 'heg':
        config = vars(heg_args())


    torch.cuda.set_device(torch.device(config['device']))

    # 训练好的模型存储路径
    check_point_name = config['backbone']+'_'+config['loss']
    checkpoints_dir = os.path.join('checkpoints', config['dataset'], check_point_name)
    if config['hyperparameter']:
        para_list = []
        folder_name = ''
        for para in config['hyperparameter']:
            if isinstance(config[para], list):
                value = config[para][-1]
            else:
                value = config[para]

            if isinstance(value, str):
                value = value
            else:
                value = str(value)

            para_list.append(para + '_' + value)
            folder_name = '_'.join(para_list)
    else:
        folder_name = 'base'
    checkpoints_dir = os.path.join(checkpoints_dir, folder_name)



    os.makedirs(checkpoints_dir, exist_ok=True)

    # 打印所有参数
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    # 生成config.yml
    with open(os.path.join(checkpoints_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    # 选择backbone
    print("=> creating model %s" % config['backbone'])
    print("=> creating loss %s" % config['loss'])



    # 载入训练、验证集
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

    # 设置dataset和dataloader
    train_dataset = Dataset[config['dataset']](
        config=config,
        img_ids=train_img_ids,
        num_classes=config['num_classes'],
        lens=lens,
        mode='train'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=config['num_workers'],
        drop_last=True)

    val_dataset = Dataset[config['dataset']](
        config=config,
        img_ids=val_img_ids,
        num_classes=config['num_classes'],
        lens=lens,
        mode='val'
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config['num_workers'],
        drop_last=True)


    cudnn.benchmark = True
    # 选择backbone
    backbones = {
        'unet': UNet,
        'nested_unet': NestedUNet,
        'att_unet': AttU_Net,
        'res_unet': ResUnet,
        'res_unet_a': Res_UNet_A,
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

    if config['dataset'] == 'HCMS':
        config['bds_num_classes'] = config['num_classes']
        config['sdf_num_classes'] = config['num_classes']

    model = backbones[config['backbone']](config)
    model = model.to(config['device'])

    if config['resume'] is not None:
        model.load_state_dict(torch.load(config['resume'], map_location=lambda storage, loc: storage))


    print(f'Number of parameters: {count_params(model)}')

    # 选择loss
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

    # 选择optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['max_lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['max_lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # 选择scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config['epochs']), eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, min_lr=config['min_lr'])
    elif config['scheduler'] == 'CyclicLR':
        scheduler = lr_scheduler.CyclicLR(optimizer,
                                          base_lr=config['min_lr'], max_lr=config['max_lr'],
                                          cycle_momentum=False, step_size_up=5, step_size_down=config['epochs']-5)
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=4, eta_min=config['min_lr'])  # T_0：第一个周期，T_mult：后一个周期是前一个的几倍
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 100, 150, 200], gamma=0.5)
    else:
        scheduler = lr_scheduler.ConstantLR(optimizer)

    # 模型和log保存路径
    summary_write_dir = os.path.join('sum_wr', config['dataset'], check_point_name,
                                     time.strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(summary_write_dir)

    # 提前结束循环
    best_loss = float('inf')
    best_Iou = 0
    trigger = 0

    # 开始迭代
    for epoch in range(1, config['epochs'] + 1):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        if config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['val_iou'])
        else:
            scheduler.step()


        # log
        writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('Total_Loss/', {'Train_Loss' : train_log['train_loss'], 'Val_Loss': val_log['val_loss']}, epoch)
        writer.add_scalars('Main_Loss/', {'Train_Main_Loss' : train_log['train_main_loss'], 'Val_Main_Loss': val_log['val_main_loss']}, epoch)
        writer.add_scalars('Aux_Loss/', {'Train_Aux_Loss' : train_log['train_aux_loss'], 'Val_Aux_Loss': val_log['val_aux_loss']}, epoch)
        writer.add_scalars('Iou/', {'Train_Iou' : train_log['train_iou'], 'Val_Iou': val_log['val_iou']}, epoch)
        writer.add_scalars('F1s/', {'Train_F1s' : train_log['train_f1s'], 'Val_F1s': val_log['val_f1s']}, epoch)
        trigger += 1

        #
        if (epoch % 40) == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model' + str(epoch) + '.pth'))

        # 保存val_loss最低的一代
        if val_log['val_iou'] > best_Iou:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'model.pth'))
            best_Iou = val_log['val_iou']
            print("=> saved best model")
            print('Best_Iou:', best_Iou)
            print('-' * 20)
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break
        torch.cuda.empty_cache()
    writer.close()


if __name__ == '__main__':
    main()
