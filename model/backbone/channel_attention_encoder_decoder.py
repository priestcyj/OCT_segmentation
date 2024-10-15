import torch
from torch import nn
from model.backbone.channel_attention_block import ChannelAttentionBlock


class ChannelAttentionEncoderBlock(nn.Module):
    def __init__(self, config, nb_filter):
        super(ChannelAttentionEncoderBlock, self).__init__()
        self.config = config
        self.input_channels = config['input_channels']
        self.enc_conv0 = ChannelAttentionBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv1 = ChannelAttentionBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = ChannelAttentionBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = ChannelAttentionBlock(nb_filter[2], nb_filter[3], nb_filter[3])

    def forward(self, x):
        enc0, enc_att0 = self.enc_conv0(x)
        enc1, enc_att1 = self.enc_conv1(self.pool1(enc0))
        enc2, enc_att2 = self.enc_conv2(self.pool2(enc1))
        x, enc_att3 = self.enc_conv3(self.pool3(enc2))

        enc_list = [enc0, enc1, enc2]
        enc_att_list = {'enc_att0': enc_att0, 'enc_att1': enc_att1, 'enc_att2': enc_att2, 'enc_att3': enc_att3}
        return x, enc_list, enc_att_list


class ChannelAttentionDecoderConnectBlock(nn.Module):
    def __init__(self, config, nb_filter):
        super(ChannelAttentionDecoderConnectBlock, self).__init__()
        self.config = config
        self.num_classes = config['num_classes']

        self.dec_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv1 = ChannelAttentionBlock(nb_filter[3] * 2 + nb_filter[2], nb_filter[2], nb_filter[2])
        self.bou_dec_conv1 = ChannelAttentionBlock(nb_filter[3] * 2 + nb_filter[2], nb_filter[2], nb_filter[2])

        self.dec_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv2 = ChannelAttentionBlock(nb_filter[2] * 2 + nb_filter[1], nb_filter[1], nb_filter[1])
        self.bou_dec_conv2 = ChannelAttentionBlock(nb_filter[2] * 2 + nb_filter[1], nb_filter[1], nb_filter[1])

        self.dec_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv3 = ChannelAttentionBlock(nb_filter[1] * 2 + nb_filter[0], nb_filter[0], nb_filter[0])
        self.bou_dec_conv3 = ChannelAttentionBlock(nb_filter[1] * 2 + nb_filter[0], nb_filter[0], nb_filter[0])

        self.seg_final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        self.bou_final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)

    def forward(self, x, enc_output):
        seg_dec1, seg_dec_att1 = self.seg_dec_conv1(torch.cat([enc_output[2], self.dec_up1(x), self.dec_up1(x)], 1))
        bou_dec1, bou_dec_att1 = self.bou_dec_conv1(torch.cat([enc_output[2], self.dec_up1(x), self.dec_up1(x)], 1))

        seg_dec2, seg_dec_att2 = self.seg_dec_conv2(torch.cat([enc_output[1], self.dec_up2(seg_dec1), self.dec_up2(bou_dec1)], 1))
        bou_dec2, bou_dec_att2 = self.bou_dec_conv2(torch.cat([enc_output[1], self.dec_up2(seg_dec1), self.dec_up2(bou_dec1)], 1))

        seg_dec3, seg_dec_att3 = self.seg_dec_conv3(torch.cat([enc_output[0], self.dec_up3(seg_dec2), self.dec_up3(bou_dec2)], 1))
        bou_dec3, bou_dec_att3 = self.bou_dec_conv3(torch.cat([enc_output[0], self.dec_up3(seg_dec2), self.dec_up3(bou_dec2)], 1))

        seg_out = self.seg_final(seg_dec3)
        bou_out = self.bou_final(bou_dec3)

        return seg_out, bou_out





