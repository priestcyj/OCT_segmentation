import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataset.aroi_dataset import AROI
from dataset.duke_dataset import DUKE
from dataset.hcms_dataset import HCMS
from dataset.heg_dataset import HEG
from losses import *
from train import train, validate


def tuning_function(config, pruning_config, model, writer, iteration=0):
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
    optimizer = optim.Adam(params, lr=config['max_lr'], weight_decay=config['weight_decay'])
    # scheduler = lr_scheduler.ConstantLR(optimizer)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=config['min_lr'], max_lr=config['max_lr'], cycle_momentum=False, step_size_up=5, step_size_down=55)



    # 提前结束循环
    trigger = 0
    best_Iou = 0

    # 开始迭代
    if iteration == pruning_config['num_iterations']:
        tuning_epochs = pruning_config['tuning_epochs']*12
    else:
        tuning_epochs = pruning_config['tuning_epochs']
    for epoch in range(1, tuning_epochs+1):
        print('Epoch [%d/%d]' % (epoch, pruning_config['tuning_epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        scheduler.step()

        # log
        writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalars('Total_Loss/', {'Train_Loss': train_log['train_loss'], 'Val_Loss': val_log['val_loss']},
                           epoch)
        writer.add_scalars('Main_Loss/',
                           {'Train_Main_Loss': train_log['train_main_loss'], 'Val_Main_Loss': val_log['val_main_loss']},
                           epoch)
        writer.add_scalars('Aux_Loss/',
                           {'Train_Aux_Loss': train_log['train_aux_loss'], 'Val_Aux_Loss': val_log['val_aux_loss']},
                           epoch)
        writer.add_scalars('Iou/', {'Train_Iou': train_log['train_iou'], 'Val_Iou': val_log['val_iou']}, epoch)
        writer.add_scalars('F1s/', {'Train_F1s': train_log['train_f1s'], 'Val_F1s': val_log['val_f1s']}, epoch)
        trigger += 1

        # 保存val_loss最低的一代
        if val_log['val_iou'] > best_Iou:
            torch.save(model, os.path.join(pruning_config['checkpoints'], ('model_tuned_iter_' + str(iteration))))
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

    del optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    return model








