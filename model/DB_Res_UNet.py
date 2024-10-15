import torch
from torch import nn

from model.DB_CA_UNet import ResChannelAttentionEncoderBlock, ResidualChannelAttentionBlock
from model.backbone.encoder_block import UNetEncoderBlock
from model.backbone.decoder_block import UNetDecoderBlock, UNetDecoderConnectBlock
from model.backbone.projection import ProjectionHead


class DB_Res_UNet(nn.Module):
    def __init__(self, config, is_connect=False, **kwargs):
        super().__init__()
        self.is_connect = is_connect

        self.num_classes = config['num_classes']
        self.bds_num_classes = config['bds_num_classes']

        self.input_channels = config['input_channels']

        nb_filter = config['nb_filters']

        self.encoder = ResChannelAttentionEncoderBlock(config, nb_filter, is_skip=True)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if self.is_connect:
            self.decoder = ResDecoderConnectBlock(config, nb_filter)
        else:
            self.seg_decoder = ResDecoderBlock(config, nb_filter)
            self.bou_decoder = ResDecoderBlock(config, nb_filter)

        self.seg_final = ProjectionHead(nb_filter[0], self.num_classes, proj='res')
        if 'br' in config['loss']:
            self.dbs_final = ProjectionHead(nb_filter[0], self.bds_num_classes, proj='res')
        elif 'sdf' in config['loss']:
            self.dbs_final = ProjectionHead(nb_filter[0], self.num_classes, proj='res')

    def forward(self, x):
        x, enc, _ = self.encoder(x)

        if self.is_connect:
            seg_dec, bou_dec = self.decoder(x, enc)
        else:
            seg_dec = self.seg_decoder(x, enc)
            bou_dec = self.bou_decoder(x, enc)

        seg_out = self.seg_final(seg_dec)
        dbs_out = self.dbs_final(bou_dec)
        
        return {'segment_output': seg_out, 'boundary_output': dbs_out}

class ResDecoderConnectBlock(nn.Module):
    def __init__(self, config, nb_filter, is_skip=True):
        super(ResDecoderConnectBlock, self).__init__()
        self.config = config
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']
        
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_block0 = ResidualChannelAttentionBlock(nb_filter[4]*2+nb_filter[3], nb_filter[3],1, 1, is_skip)
        self.bou_block0 = ResidualChannelAttentionBlock(nb_filter[4]*2+nb_filter[3], nb_filter[3],1, 1, is_skip)

        self.seg_block1 = ResidualChannelAttentionBlock(nb_filter[3]*2+nb_filter[2], nb_filter[2],1, 1, is_skip)
        self.bou_block1 = ResidualChannelAttentionBlock(nb_filter[3]*2+nb_filter[2], nb_filter[2],1, 1, is_skip)

        self.seg_block2 = ResidualChannelAttentionBlock(nb_filter[2]*2+nb_filter[1], nb_filter[1],1, 1, is_skip)
        self.bou_block2 = ResidualChannelAttentionBlock(nb_filter[2]*2+nb_filter[1], nb_filter[1],1, 1, is_skip)

        self.seg_block3 = ResidualChannelAttentionBlock(nb_filter[1]*2+nb_filter[0], nb_filter[0],1, 1, is_skip)
        self.bou_block3 = ResidualChannelAttentionBlock(nb_filter[1]*2+nb_filter[0], nb_filter[0],1, 1, is_skip)

  

    def forward(self, x, enc_output):

        seg_dec0, seg_att0 = self.seg_block0(torch.cat([enc_output[3], self.up(x), self.up(x)], 1))
        bou_dec0, bou_att0 = self.bou_block0(torch.cat([enc_output[3], self.up(x), self.up(x)], 1))

        seg_dec1, seg_att1 = self.seg_block1(torch.cat([enc_output[2], self.up(seg_dec0), self.up(bou_dec0)], 1))
        bou_dec1, bou_att1 = self.bou_block1(torch.cat([enc_output[2], self.up(seg_dec0), self.up(bou_dec0)], 1))

        seg_dec2, seg_att2 = self.seg_block2(torch.cat([enc_output[1], self.up(seg_dec1), self.up(bou_dec1)], 1))
        bou_dec2, bou_att2 = self.bou_block2(torch.cat([enc_output[1], self.up(seg_dec1), self.up(bou_dec1)], 1))

        seg_dec3, seg_att3 = self.seg_block3(torch.cat([enc_output[0], self.up(seg_dec2), self.up(bou_dec2)], 1))
        bou_dec3, bou_att3 = self.bou_block3(torch.cat([enc_output[0], self.up(seg_dec2), self.up(bou_dec2)], 1))




        return seg_dec3, bou_dec3


class ResDecoderBlock(nn.Module):
    def __init__(self, config, nb_filter, is_skip=True):
        super(ResDecoderBlock, self).__init__()
        self.config = config
        self.input_channels = config['input_channels']
        self.num_classes = config['num_classes']

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_block0 = ResidualChannelAttentionBlock(nb_filter[4] + nb_filter[3], nb_filter[3], 1, 1, is_skip)
        self.res_block1 = ResidualChannelAttentionBlock(nb_filter[3] + nb_filter[2], nb_filter[2], 1, 1, is_skip)
        self.res_block2 = ResidualChannelAttentionBlock(nb_filter[2] + nb_filter[1], nb_filter[1], 1, 1, is_skip)
        self.res_block3 = ResidualChannelAttentionBlock(nb_filter[1] + nb_filter[0], nb_filter[0], 1, 1, is_skip)

    def forward(self, x, enc_output):
        res_dec0, res_att0 = self.res_block0(torch.cat([enc_output[3], self.up(x)], 1))
        res_dec1, res_att1 = self.res_block1(torch.cat([enc_output[2], self.up(res_dec0)], 1))
        res_dec2, res_att2 = self.res_block2(torch.cat([enc_output[1], self.up(res_dec1)], 1))
        res_dec3, res_att3 = self.res_block3(torch.cat([enc_output[0], self.up(res_dec2)], 1))
        return res_dec3


class DB_Res_UNet_ConnectTrue(DB_Res_UNet):
    def __init__(self, config, **kwargs):
        super().__init__(config, is_connect=True, **kwargs)

class DB_Res_UNet_ConnectFalse(DB_Res_UNet):
    def __init__(self, config, **kwargs):
        super().__init__(config, is_connect=False, **kwargs)
