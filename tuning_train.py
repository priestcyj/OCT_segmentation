
import os
import time
import yaml
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

from pruning.tuning_function import tuning_function
from utils import count_params


def pruning_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints',
                        default=r'checkpoints/HCMS/res_db_unet_cross_attention_with_channel_attention_dice_br_sdf_loss')
    parser.add_argument('--num_iterations', default=1, help='Number of Recovery Epochs')
    parser.add_argument('--tuning_epochs', default=60, help='Number of Recovery Epochs')
    args = parser.parse_args()
    return args

def main():
    # 相关参数输入
    pruning_config = vars(pruning_args())

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

    model_path = os.path.join(pruning_config['checkpoints'], ('model_pruned_iter_' + str(pruning_config['num_iterations'])))
    # 选择backbone

    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = model.cuda()
    print(f'Number of parameters: {count_params(model)}')



    # start pruning & tuning
    tuning_function(config, pruning_config, model, writer, iteration=pruning_config['num_iterations'])

if __name__ == '__main__':
    main()


