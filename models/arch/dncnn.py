import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17, **kwargs):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class YTMTBlock(nn.Module):
    def __init__(self):
        super(YTMTBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), input_l - self.relu(input_l)
        out_rp, out_rn = self.relu(input_r), input_r - self.relu(input_r)
        out_l = out_lp + out_rn
        out_r = out_rp + out_ln
        return out_l, out_r


class ConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1, bias=True, dilation=1):
        super(ConvBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
        )

    def forward(self, x):
        return self.model(x)


class YTMTConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, skip=False, **kwargs):
        super(YTMTConvBlock, self).__init__()
        self.conv_l = ConvBlock(in_channels, out_channels, **kwargs)
        self.conv_r = ConvBlock(in_channels, out_channels, **kwargs)
        self.ytmt_norm = YTMTBlock()
        self.skip = skip

    def forward(self, input_l, input_r):
        out_l = self.conv_l(input_l)
        out_r = self.conv_r(input_r)
        out_l, out_r = self.ytmt_norm(out_l, out_r)

        if self.skip and input_l.shape == out_l.shape and input_r.shape == out_r.shape:
            out_l += input_l
            out_r += input_r
        return out_l, out_r


class AttConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1, bias=True, dilation=1):
        super(AttConvBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        self.model = nn.Sequential(
            PALayer(in_channels),
            CALayer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
        )

    def forward(self, x):
        return self.model(x)


class YTMTAttConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, skip=False, **kwargs):
        super(YTMTAttConvBlock, self).__init__()
        self.conv_l = AttConvBlock(in_channels, out_channels, **kwargs)
        self.conv_r = AttConvBlock(in_channels, out_channels, **kwargs)
        self.ytmt_norm = YTMTBlock()
        self.skip = skip

    def forward(self, input_l, input_r):
        out_l = self.conv_l(input_l)
        out_r = self.conv_r(input_r)
        out_l, out_r = self.ytmt_norm(out_l, out_r)

        if self.skip and input_l.shape == out_l.shape and input_r.shape == out_r.shape:
            out_l += input_l
            out_r += input_r
        return out_l, out_r


class YTMTOutBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1, bias=False):
        super(YTMTOutBlock, self).__init__()
        padding = padding or (kernel_size - 1) // 2
        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias),
        )

        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias),
        )

    def forward(self, input_l, input_r):
        out_l = self.conv_l(input_l)
        out_r = self.conv_r(input_r)
        return out_l, out_r


class YTMTDnCNN(nn.Module):
    def __init__(self, channels, inter_channels=64, num_of_layers=10):
        super(YTMTDnCNN, self).__init__()
        kernel_size = 3
        features = inter_channels
        convs = []

        self.nums_of_layers = num_of_layers

        self.conv1 = YTMTConvBlock(channels, features, kernel_size=kernel_size)

        for i in range(num_of_layers - 2):
            convs.append(YTMTAttConvBlock(features, features, kernel_size=kernel_size))

        self.convs = nn.Sequential(*convs)
        self.out = YTMTOutBlock(features, channels)

    def forward(self, x, y=None, fn=None):
        out_l, out_r = self.conv1(x, y if y is not None else x)
        for conv in self.convs:
            out_l, out_r = conv(out_l, out_r)
        out_l, out_r = self.out(out_l, out_r)
        return out_l, out_r


if __name__ == '__main__':
    pass
