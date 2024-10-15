import torch
from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class UNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        input_channels = config['input_channels']
        num_classes = config['num_classes']

        nb_filter = config['nb_filters']

        self.enc_conv0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv1 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        self.enc_conv4 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])



    def forward(self, x):
        enc0 = self.enc_conv0(x)
        enc1 = self.enc_conv1(self.pool1(enc0))
        enc2 = self.enc_conv2(self.pool2(enc1))
        enc3 = self.enc_conv3(self.pool3(enc2))
        enc4 = self.enc_conv4(self.pool4(enc3))

        dec1 = self.dec_conv1(torch.cat([enc3, self.up2(enc4)], 1))
        dec2 = self.dec_conv2(torch.cat([enc2, self.up3(dec1)], 1))
        dec3 = self.dec_conv3(torch.cat([enc1, self.up4(dec2)], 1))
        dec4 = self.dec_conv4(torch.cat([enc0, self.up4(dec3)], 1))

        output = dec4

        return output


