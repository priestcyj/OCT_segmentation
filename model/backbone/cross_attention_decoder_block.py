import torch
from torch import nn
from model.backbone.channel_attention_block import ChannelAttentionBlock, ChannelAttention


class CrossAttentionDecoderBlock(nn.Module):
    def __init__(self, config, nb_filter):
        super(CrossAttentionDecoderBlock, self).__init__()
        self.config = config
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']

        self.seg_conv0 = SingleChannelAttentionBlock(nb_filter[3], nb_filter[2])
        self.bou_conv0 = SingleChannelAttentionBlock(nb_filter[3], nb_filter[2])



        self.seg_cross_block1 = CrossAttentionBlock(nb_filter[2], nb_filter[2], nb_filter[2])
        self.bou_cross_block1 = CrossAttentionBlock(nb_filter[2], nb_filter[2], nb_filter[2])

        self.seg_cross_block2 = CrossAttentionBlock(nb_filter[1], nb_filter[1], nb_filter[1])
        self.bou_cross_block2 = CrossAttentionBlock(nb_filter[1], nb_filter[1], nb_filter[1])

        self.seg_cross_block3 = CrossAttentionBlock(nb_filter[0], nb_filter[0], nb_filter[0])
        self.bou_cross_block3 = CrossAttentionBlock(nb_filter[0], nb_filter[0], nb_filter[0])

        self.seg_final = nn.Conv2d(nb_filter[0]//2, self.num_classes, kernel_size=1)
        self.bou_final = nn.Conv2d(nb_filter[0]//2, self.num_classes, kernel_size=1)

    def forward(self, x, enc_output):
        seg_dec0, seg_att0 = self.seg_conv0(x)
        bou_dec0, bou_att0 = self.bou_conv0(x)

        seg_dec1, seg_att1 = self.seg_cross_block1(seg_dec0, bou_dec0, enc_output[-1])
        bou_dec1, bou_att1 = self.bou_cross_block1(bou_dec0, seg_dec0, enc_output[-1])
        

        seg_dec2, seg_att2 = self.seg_cross_block2(seg_dec1, bou_dec1, enc_output[-2])
        bou_dec2, bou_att2 = self.bou_cross_block2(bou_dec1, seg_dec1, enc_output[-2])
        

        seg_dec3, seg_att3 = self.seg_cross_block3(seg_dec2, bou_dec2, enc_output[-3])
        bou_dec3, bou_att3 = self.bou_cross_block3(bou_dec2, seg_dec2, enc_output[-3])

        seg_out = self.seg_final(seg_dec3)
        bou_out = self.bou_final(bou_dec3)

        seg_att_dict = {'seg_att1': seg_att1, 'seg_att2': seg_att2, 'seg_att3': seg_att3}

        bou_att_dict = {'bou_att1': bou_att1, 'bou_att2': bou_att2, 'bou_att3': bou_att3}

        return seg_out, bou_out, seg_att_dict, bou_att_dict


class CrossSpatialAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(CrossSpatialAttention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class CrossAttentionBlock(nn.Module):
    def __init__(self, main_channels, aux_channels, enc_channels):
        super(CrossAttentionBlock, self).__init__()

        self.dec_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 在aux上做att
        self.aux_att = CrossSpatialAttention(F_g=main_channels, F_l=aux_channels, F_int=aux_channels//2)
        self.enc_att = CrossSpatialAttention(F_g=main_channels, F_l=enc_channels, F_int=enc_channels//2)
        self.dec_conv = ChannelAttentionBlock(main_channels+aux_channels+enc_channels, main_channels, main_channels//2)
        

    def forward(self, main_feature, aux_feature, enc_feature):
        main_feature = self.dec_up(main_feature)
        aux_feature = self.dec_up(aux_feature)
        aux_feature = self.aux_att(g=main_feature, x=aux_feature)
        enc_feature = self.enc_att(g=main_feature, x=enc_feature)
        dec, dec_att = self.dec_conv(torch.cat([main_feature, aux_feature, enc_feature],1))
        return dec, dec_att


class SingleChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.aat = ChannelAttention(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out, att = self.aat(out)
        out = self.bn(out)
        out = self.relu(out)

        return out, {'att': att}
