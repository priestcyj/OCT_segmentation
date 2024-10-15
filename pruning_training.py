import os
import time
import numpy as np
import yaml
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from pruning.pruning_function import pruning_function
from pruning.tuning_function import tuning_function
from utils import count_params
from model import *


def pruning_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints',
                        default=r'checkpoints/HEG/res_db_unet_cross_attention_with_channel_attention_dice_weight_sdf_loss/num_KFold_kf_1_nb_filters_256')
    parser.add_argument('--resume', default=None, help='Number of Recovery Epochs')


    parser.add_argument('--pruning_mode', default='attention', help='pruning mode')
    parser.add_argument('--num_iterations', default=10, help='Number of interation')
    parser.add_argument('--pruning_percentage', default=0.1, help='pruning percentage')
    parser.add_argument('--pretrain_epochs', default=5)
    parser.add_argument('--tuning_epochs', default=5, help='Number of Recovery Epochs')


    args = parser.parse_args()
    return args

def main():
    # 相关参数输入
    pruning_config = vars(pruning_args())

    # 打印所有参数
    print('-' * 20)
    for key in pruning_config:
        print('%s: %s' % (key, pruning_config[key]))
    print('-' * 20)

    # 生成pruning_config.yml
    with open(os.path.join(pruning_config['checkpoints'], 'pruning_config.yml'), 'w') as f:
        yaml.dump(pruning_config, f)



    # ------------------------读取config------------------------ #
    config_path = os.path.join(pruning_config['checkpoints'], 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    # 选择backbone
    print("=> creating model %s" % config['backbone'])
    print("=> creating loss %s" % config['loss'])

    # 模型和log保存路径
    writer = SummaryWriter(os.path.join('sum_wr',
                                        config['backbone'] + '_' + config['loss'] + '_' + config['num_KFold'],
                                        time.strftime("%Y-%m-%d_%H-%M-%S")))

    if pruning_config['resume'] is None:
        model_path = os.path.join(pruning_config['checkpoints'], 'model.pth')# 选择backbone
        backbones = {
            'res_db_unet_cross_attention':
                ResDoubleBranchCrossAttentionUNet,
            'res_db_unet_cross_attention_with_channel_attention':
                ResDoubleBranchCrossAttentionUNetWithChannelAttention
        }
        model = backbones[config['backbone']](config)
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        model = model.to(config['device'])
    else:
        model_path = os.path.join(pruning_config['checkpoints'], 'model_' + pruning_config['resume'])
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = model.to(config['device'])



    # start pruning & tuning

    for iteration in range(1, pruning_config['num_iterations'] + 1):
        # 评价模型卷积核的重要性并对模型进行裁剪
        pruning_function(config, pruning_config, iteration, model)
        torch.cuda.empty_cache()

        pruned_best_model_path = os.path.join(pruning_config['checkpoints'], ('model_pruned_iter_' + str(iteration)))
        model = torch.load(pruned_best_model_path, map_location=lambda storage, loc: storage)
        model = model.to(config['device'])

        print('-' * 20)
        print('Pruning completed.')
        print(f'Number of parameters: {count_params(model)}')

        # 裁剪后对模型进行微调
        model = tuning_function(config, pruning_config, model, writer, iteration)



        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


