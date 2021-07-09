import torch
from torch import nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DBFPN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels

        # reduce layers
        self.in2_conv = ConvBnRelu(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1)
        self.in3_conv = ConvBnRelu(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1)
        self.in4_conv = ConvBnRelu(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1)
        self.in5_conv = ConvBnRelu(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1)

        # smooth layers
        self.p5_conv = ConvBnRelu(in_channels=self.out_channels, out_channels=self.out_channels // 4,
                                  kernel_size=3, padding=1)
        self.p4_conv = ConvBnRelu(in_channels=self.out_channels, out_channels=self.out_channels // 4,
                                  kernel_size=3, padding=1)
        self.p3_conv = ConvBnRelu(in_channels=self.out_channels, out_channels=self.out_channels // 4,
                                  kernel_size=3, padding=1)
        self.p2_conv = ConvBnRelu(in_channels=self.out_channels, out_channels=self.out_channels // 4,
                                  kernel_size=3, padding=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def upsample_add(x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True) + y

    def forward(self, x):
        c2, c3, c4, c5 = x

        # 横向连接
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        # 上采样+叠加
        out4 = self.upsample_add(in5, in4)
        out3 = self.upsample_add(out4, in3)
        out2 = self.upsample_add(out3, in2)

        # smooth
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        # 上采样+融合
        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        return torch.cat([p5, p4, p3, p2], dim=1)
