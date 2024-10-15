import argparse

from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',
                        default=None,
                        type=str)

    parser.add_argument('--hyperparameter', default=['sdf_out_max', 'sdf_tau'],
                        type=list)


    parser.add_argument('--device', default="cuda:0", help='device')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num_KFold', default= 'kf_1', type=str, metavar='N',
                        help='number of KFold')
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='mini-batch size (default: 16)')


    # ----------------------------------------model----------------------------------------#
    parser.add_argument('--backbone', metavar='ARCH',
                        default='res_db_unet_cross_attention')
    # unet, nested_unet, att_unet, res_unet, transunet


    # dh_res_unet
    # db_res_unet_connect_false, db_res_unet_connect_true

    # res_db_unet_cross_attention, res_db_unet_cross_attention_with_channel_attention



    # ----------------------------------------losses----------------------------------------#
    parser.add_argument('--loss', default='dice_sdf_loss')
    # dice_loss
    # dice_br_loss

    # dice_sdf_loss
    # dice_weight_sdf_loss
    # ce_dice_weight_sdf_loss

    parser.add_argument('--ce_weight',
                        default=[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
                        type=list)
    parser.add_argument('--ce_lamda', default=0.5, type=float)
    parser.add_argument('--dice_lamda', default=1, type=float)

    parser.add_argument('--br_lamda', default=0.5, type=float)
    parser.add_argument('--br_tau', default=1, type=float)


    parser.add_argument('--sdf_lamda', default=1, type=float)

    parser.add_argument('--sdf_out_max', default=50, type=float)
    parser.add_argument('--sdf_tau', default=10, type=float)


    # ----------------------------------------model_param----------------------------------------#
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=9, type=int,
                        help='number of classes')
    parser.add_argument('--nb_filters', default=[32, 64, 96, 128, 256], type=list,
                        help='number of filters')
    # [32, 32, 32, 32, 32]
    # [32, 64, 128, 128, 128]
    # [32, 64, 96, 128, 256]
    # [32, 64, 128, 256, 512]

    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--connection_scheme', default=False)  # False, True



    # ----------------------------------------dataset----------------------------------------#
    parser.add_argument('--dataset', default='HCMS',
                        help='dataset name')  # IBSR, AROI



    # ----------------------------------------optimizer----------------------------------------#
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='losses: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--max_lr', default=1e-3, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')


    # ----------------------------------------scheduler----------------------------------------#
    parser.add_argument('--scheduler', default='CyclicLR',
                        choices=['ConstantLR', 'MultiStepLR', 'CosineAnnealingLR',
                                 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CyclicLR'])
    parser.add_argument('--early_stopping', default=20, type=int, metavar='N',
                        help='early stopping (default: -1)')
    parser.add_argument('--num_workers', default=2, type=int)

    config = parser.parse_args()

    return config
