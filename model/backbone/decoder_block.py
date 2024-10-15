import torch
from torch import nn
from model.backbone.vgg_block import VGGBlock


class UNetDecoderBlock(nn.Module):
    def __init__(self, config):
        super(UNetDecoderBlock, self).__init__()
        self.config = config
        self.num_classes = config['num_classes']
        nb_filter = config['nb_filters']

        self.dec_up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv0 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.dec_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.dec_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.dec_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)

    def forward(self, x, enc_output):
        dec0 = self.dec_conv0(torch.cat([enc_output[3], self.dec_up1(x)], 1))
        dec1 = self.dec_conv1(torch.cat([enc_output[2], self.dec_up1(dec0)], 1))
        dec2 = self.dec_conv2(torch.cat([enc_output[1], self.dec_up2(dec1)], 1))
        dec3 = self.dec_conv3(torch.cat([enc_output[0], self.dec_up3(dec2)], 1))
        out = self.final(dec3)
        return out


class UNetDecoderConnectBlock(nn.Module):
    def __init__(self, config):
        super(UNetDecoderConnectBlock, self).__init__()
        self.config = config
        self.num_classes = config['num_classes']
        nb_filter = config['nb_filters']

        self.dec_up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv0 = VGGBlock(nb_filter[4] * 2 + nb_filter[3], nb_filter[3], nb_filter[3])
        self.bou_dec_conv0 = VGGBlock(nb_filter[4] * 2 + nb_filter[3], nb_filter[3], nb_filter[3])
        
        self.dec_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv1 = VGGBlock(nb_filter[3] * 2 + nb_filter[2], nb_filter[2], nb_filter[2])
        self.bou_dec_conv1 = VGGBlock(nb_filter[3] * 2 + nb_filter[2], nb_filter[2], nb_filter[2])

        self.dec_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv2 = VGGBlock(nb_filter[2] * 2 + nb_filter[1], nb_filter[1], nb_filter[1])
        self.bou_dec_conv2 = VGGBlock(nb_filter[2] * 2 + nb_filter[1], nb_filter[1], nb_filter[1])

        self.dec_up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_dec_conv3 = VGGBlock(nb_filter[1] * 2 + nb_filter[0], nb_filter[0], nb_filter[0])
        self.bou_dec_conv3 = VGGBlock(nb_filter[1] * 2 + nb_filter[0], nb_filter[0], nb_filter[0])

        self.seg_final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        self.bou_final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)

    def forward(self, x, enc_output):

        seg_dec0 = self.seg_dec_conv0(torch.cat([enc_output[3], self.dec_up0(x), self.dec_up0(x)], 1))
        bou_dec0 = self.bou_dec_conv0(torch.cat([enc_output[3], self.dec_up0(x), self.dec_up0(x)], 1))

        seg_dec1 = self.seg_dec_conv1(torch.cat([enc_output[2], self.dec_up1(seg_dec0), self.dec_up1(bou_dec0)], 1))
        bou_dec1 = self.bou_dec_conv1(torch.cat([enc_output[2], self.dec_up1(seg_dec0), self.dec_up1(bou_dec0)], 1))
        
        seg_dec2 = self.seg_dec_conv2(torch.cat([enc_output[1], self.dec_up2(seg_dec1), self.dec_up2(bou_dec1)], 1))
        bou_dec2 = self.bou_dec_conv2(torch.cat([enc_output[1], self.dec_up2(seg_dec1), self.dec_up2(bou_dec1)], 1))

        seg_dec3 = self.seg_dec_conv3(torch.cat([enc_output[0], self.dec_up3(seg_dec2), self.dec_up3(bou_dec2)], 1))
        bou_dec3 = self.bou_dec_conv3(torch.cat([enc_output[0], self.dec_up3(seg_dec2), self.dec_up3(bou_dec2)], 1))

        seg_out = self.seg_final(seg_dec3)
        bou_out = self.bou_final(bou_dec3)

        return seg_out, bou_out


