import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualBlock, self).__init__()

        self.bn0 = nn.BatchNorm2d(input_dim)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

        self.conv_skip = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1)
        self.bn_skip = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x0 = self.bn0(x)
        x0 = self.relu0(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)

        x_skip = self.conv_skip(x)
        x_skip = self.bn_skip(x_skip)
        output = x2+x_skip

        return output


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()


        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
        elif proj == 'res':
            self.proj = nn.Sequential(
                ResidualBlock(dim_in, dim_in//2, 1, 1),
                nn.Conv2d(dim_in//2, proj_dim, kernel_size=1),
            )


    def forward(self, x):
        return self.proj(x)


