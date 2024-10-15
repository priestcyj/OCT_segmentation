import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算输入特征图的平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积层生成空间注意力图
        attention = self.sigmoid(self.conv(concat))

        # 将空间注意力图应用于输入特征图
        out = x * attention

        return out, attention




# 测试
if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 32, 32)  # 例子输入: Batch size 1, 64 channels, 32x32 feature map
    spatial_attention = SpatialAttention(kernel_size=7)
    output, attention = spatial_attention(input_tensor)
    print(output.size(), attention.size())
