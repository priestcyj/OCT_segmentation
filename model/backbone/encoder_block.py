import torch
from torch import nn
from model.backbone.vgg_block import VGGBlock


class UNetEncoderBlock(nn.Module):
    def __init__(self, config):
        super(UNetEncoderBlock, self).__init__()
        self.input_channels = config['input_channels']
        nb_filter = config['nb_filters']

        self.enc_conv0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv1 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc_conv4 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

    def forward(self, x):
        enc0 = self.enc_conv0(x)
        enc1 = self.enc_conv1(self.pool1(enc0))
        enc2 = self.enc_conv2(self.pool2(enc1))
        enc3 = self.enc_conv3(self.pool3(enc2))
        x = self.enc_conv4(self.pool4(enc3))
        return x, [enc0, enc1, enc2, enc3]



