import os
import torch
from tqdm import tqdm
import numpy as np
import copy
from torch.optim import lr_scheduler
import pandas as pd
import torch.optim as optim
from collections import OrderedDict
import torch.backends.cudnn as cudnn

import torch_pruning as tp

from dataset.aroi_dataset import AROI
from dataset.duke_dataset import DUKE
from dataset.heg_dataset import HEG
from train import train, validate

from model import *
from losses import *
from dataset.hcms_dataset import HCMS
from utils import count_params, AverageMeter, show_histogram


def compare_weights(model_before, model_after):
    print('-'*150)
    for (name_before, param_before), (name_after, param_after) in zip(
            model_before.named_parameters(), model_after.named_parameters()):
        if name_before != name_after:
            print(f"权重名称不同: {name_before} vs {name_after}")
        elif param_before.shape != param_after.shape:
            print(f"权重形状不同: {name_before} (裁剪前: {param_before.shape}, 裁剪后: {param_after.shape})")


def pruning_function(config, pruning_config, iteration, model):
    print('-' * 20)
    print('Pruning Iteration: ', iteration)


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
        batch_size=1,
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)

    cudnn.benchmark = True



    pbar = tqdm(total=len(train_loader))
    mean_att_dict = {}

    for images, labels, _ in train_loader:
        # if len(mean_att_dict) !=0:
        #     break
        images = images.to(config['device'], dtype=torch.float32)
        # compute output
        with torch.no_grad():
            outputs = model(images)
        att_dict = model.att_list

        for branch_key in att_dict:
            branch_dict = att_dict[branch_key]
            for layer_key in branch_dict:
                layer_dict = branch_dict[layer_key]
                for block_key in layer_dict:
                    att_value = layer_dict[block_key].detach().cpu().numpy().reshape(-1)

                    new_key = branch_key + '_' + layer_key + '_' + block_key
                    if new_key not in mean_att_dict:
                        mean_att_dict[new_key] = AverageMeter()

                    mean_att_dict[new_key].update(att_value, 1)
        del images, outputs, att_dict, att_value
        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
    # show_histogram
    # for key in mean_att_dict:
    #     show_histogram(mean_att_dict[key].avg, key)

    # model_before = ResDoubleBranchCrossAttentionUNetWithChannelAttention(config)
    # state_dict = model.state_dict()
    # model_before.load_state_dict(state_dict)

    threshold = 0.5
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 1, 256, 256).to(config['device']))
    for key in mean_att_dict:
        # if key != 'decoder_att_seg_cross_block0.dec_conv_att_conv_skip_att':
        #     continue
        key_ = key.replace('_att_', '.')
        attributes = key_.split('.')
        attributes = [n.replace('_att','') for n in attributes]
        attr = model
        for attribute in attributes:
            attr = getattr(attr, attribute)

        out_channels = attr.weight.shape[0]
        num_pruned_filters = int((out_channels*pruning_config['pruning_percentage']))
        if out_channels <= 32:
            continue
        if num_pruned_filters < 4:
            num_pruned_filters = 4
        if (out_channels - num_pruned_filters) < 32:
            num_pruned_filters = out_channels - 32 + 1
        min_idxs = np.argsort(mean_att_dict[key].avg)[:num_pruned_filters]
        threshold_idxs = np.where(mean_att_dict[key].avg <= threshold)[0]
        # 2. 获取与所剪的卷积层存在依赖的所有层，并指定需要剪枝的通道索引
        group = DG.get_pruning_group(attr, tp.prune_conv_out_channels, idxs=min_idxs)


        psi_idx = []
        psi_name = []
        for i, (dep, idx) in enumerate(group._group):
            if 'psi.' in str(dep) or len(idx) == 1:
                psi_idx.append(i)
                psi_name.append(str(dep))
            else:
                continue

        for i in psi_idx[::-1]:
            group._group.pop(i)


        # 因为堆叠产生的裁剪错误
        dep_list = []
        idx_list = []
        pop_list = []
        m = 0
        for dep, idx in group:
            if len(idx) % num_pruned_filters != 0:
                dep_list.append(dep)
                idx_list.append(idx)
                pop_list.append(m)
            m += 1

        for m in pop_list[::-1]:
            group._group.pop(m)

        for i in range(len(dep_list)):
            group.add_dep(dep=dep_list[i], idxs=idx_list[i][1:])


        # 3. 执行剪枝操作
        if len(group) != 0:
            group.prune()
        print(attributes)
        print(attr.weight.shape[0])
        torch.cuda.empty_cache()
        # compare_weights(model_before, model)


    torch.save(model, os.path.join(pruning_config['checkpoints'], ('model_pruned_iter_' + str(iteration))))





