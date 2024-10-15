import torch
from torch import nn



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 计算特征维度缩小的比例
        self.reduction_ratio = reduction_ratio
        # 用于计算特征的全局平均值和最大值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 用于计算特征的转换权重
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        # 用于将特征的转换权重应用到原始特征上
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算特征的全局平均值和最大值
        avg_value = self.avg_pool(x)

        # 将特征展平以供全连接层处理
        avg_value = avg_value.view(avg_value.size(0), -1)

        # 计算特征的转换权重
        avg_transform = self.fc(avg_value)

        # 将特征的转换权重应用到原始特征上
        attention = self.sigmoid(avg_transform).unsqueeze(2).unsqueeze(3)
        out = x * attention

        return out, attention

class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.aat1 = ChannelAttention(middle_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.aat2 = ChannelAttention(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out, conv1_att = self.aat1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out, conv2_att = self.aat2(out)
        out = self.relu2(out)

        return out, {'conv1_att': conv1_att, 'conv2_att': conv2_att}






