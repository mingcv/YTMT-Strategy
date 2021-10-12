import torch
import torch.nn as nn
import torch.nn.functional as F


############ Basic Blocks ################
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


############### YTMT Structures ##################
class AdditiveYTMTHead(nn.Module):
    def __init__(self):
        super(AdditiveYTMTHead, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), input_l - self.relu(input_l)
        out_rp, out_rn = self.relu(input_r), input_r - self.relu(input_r)
        out_l = out_lp + out_rn
        out_r = out_rp + out_ln
        return out_l, out_r


class ConcatYTMTHead(nn.Module):
    def __init__(self, channels):
        super(ConcatYTMTHead, self).__init__()
        self.relu = nn.ReLU()
        self.fusion_l = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_r = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), input_l - self.relu(input_l)
        out_rp, out_rn = self.relu(input_r), input_r - self.relu(input_r)
        out_l = self.fusion_l(torch.cat([out_lp, out_rn], dim=1))
        out_r = self.fusion_r(torch.cat([out_rp, out_ln], dim=1))
        return out_l, out_r


class PositiveYTMTHead(nn.Module):
    def __init__(self, channels):
        super(PositiveYTMTHead, self).__init__()
        self.relu = nn.ReLU()
        self.fusion_l = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_r = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, input_l, input_r):
        out_lp, out_ln = self.relu(input_l), self.relu(input_l)
        out_rp, out_rn = self.relu(input_r), self.relu(input_r)
        out_l = self.fusion_l(torch.cat([out_lp, out_rn], dim=1))
        out_r = self.fusion_r(torch.cat([out_rp, out_ln], dim=1))
        return out_l, out_r


class YTMTConvBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, skip=False, pattern='A', **kwargs):
        super(YTMTConvBlock, self).__init__()
        self.conv_l = ConvBlock(in_channels, out_channels, **kwargs)
        self.conv_r = ConvBlock(in_channels, out_channels, **kwargs)
        if pattern == 'A':
            self.ytmt_head = AdditiveYTMTHead()
        elif pattern == 'C':
            self.ytmt_head = ConcatYTMTHead(out_channels)
        elif pattern == 'P':
            self.ytmt_head = PositiveYTMTHead(out_channels)
        else:
            raise NotImplementedError()

        self.skip = skip
        self.pattern = pattern

    def forward(self, input_l, input_r):
        out_l = self.conv_l(input_l)
        out_r = self.conv_r(input_r)
        out_l, out_r = self.ytmt_head(out_l, out_r)

        if self.skip and input_l.shape == out_l.shape and input_r.shape == out_r.shape:
            out_l += input_l
            out_r += input_r
        return out_l, out_r


class YTMTAttConvBlock(YTMTConvBlock):
    def __init__(self, in_channels=3, out_channels=3, skip=False, pattern='A', **kwargs):
        super(YTMTAttConvBlock, self).__init__(in_channels, out_channels, skip, pattern=pattern, **kwargs)
        self.conv_l = AttConvBlock(in_channels, out_channels, **kwargs)
        self.conv_r = AttConvBlock(in_channels, out_channels, **kwargs)


class YTMTDownBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pattern='A', **kwargs):
        super(YTMTDownBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.model = YTMTAttConvBlock(in_channels, out_channels, pattern=pattern, **kwargs)

    def forward(self, input_l, input_r):
        out_l = self.max_pool(input_l)
        out_r = self.max_pool(input_r)
        return self.model(out_l, out_r)


class YTMTUpBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, pattern='A', **kwargs):
        super(YTMTUpBlock, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = YTMTAttConvBlock(in_channels, out_channels, pattern=pattern, **kwargs)

    def forward(self, input_l1, input_r1, input_l2, input_r2):
        out_l1 = self.up_sampling(input_l1)
        out_r1 = self.up_sampling(input_r1)

        diffY = input_l2.size()[2] - out_l1.size()[2]
        diffX = input_l2.size()[3] - out_l1.size()[3]

        out_l1 = F.pad(out_l1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

        out_r1 = F.pad(out_r1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

        out_l = torch.cat([input_l2, out_l1], dim=1)
        out_r = torch.cat([input_r2, out_r1], dim=1)

        out_l, out_r = self.model(out_l, out_r)

        return out_l, out_r


class YTMTOutBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1, bias=True, act=True):
        super(YTMTOutBlock, self).__init__()
        if isinstance(out_channels, int):
            out_channels_l = out_channels_r = out_channels
        else:
            out_channels_l, out_channels_r = out_channels

        padding = padding or (kernel_size - 1) // 2
        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_l, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias),
            nn.Sigmoid() if act else nn.Identity()
        )

        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_r, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias),
            nn.Sigmoid() if act else nn.Identity()
        )

    def forward(self, input_l, input_r):
        out_l = self.conv_l(input_l)
        out_r = self.conv_r(input_r)
        return out_l, out_r


################ YTMT Networks #####################
class YTMT_PS(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=64, num_of_layers=10, pattern='A', act=True):
        super(YTMT_PS, self).__init__()
        convs = []
        kernel_size = 3
        features = inter_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nums_of_layers = num_of_layers

        self.conv1 = YTMTConvBlock(in_channels, features, kernel_size=3, pattern=pattern)
        for i in range(num_of_layers - 2):
            convs.append(YTMTAttConvBlock(features, features, kernel_size=kernel_size, pattern=pattern))

        self.convs = nn.Sequential(*convs)

        self.out = YTMTOutBlock(features, out_channels, bias=False, act=act)

    def forward(self, x, y=None, fn=None):
        out_l, out_r = self.conv1(x, y if y is not None else x)
        for conv in self.convs:
            out_l, out_r = conv(out_l, out_r)
        out_l, out_r = self.out(out_l, out_r)
        return out_l, out_r


class YTMT_US(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pattern='A'):
        super(YTMT_US, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = YTMTConvBlock(n_channels, 64, pattern=pattern)
        self.down1 = YTMTDownBlock(64, 128, pattern=pattern)
        self.down2 = YTMTDownBlock(128, 256, pattern=pattern)
        self.down3 = YTMTDownBlock(256, 512, pattern=pattern)
        factor = 2
        self.down4 = YTMTDownBlock(512, 1024 // factor, pattern=pattern)
        self.up1 = YTMTUpBlock(1024, 512 // factor, pattern=pattern)
        self.up2 = YTMTUpBlock(512, 256 // factor, pattern=pattern)
        self.up3 = YTMTUpBlock(256, 128 // factor, pattern=pattern)
        self.up4 = YTMTUpBlock(128, 64, pattern=pattern)
        self.outc = YTMTOutBlock(64, n_classes)

    def forward(self, x, y=None, fn=None):
        x1_1, x1_2 = self.inc(x, y if y is not None else x)
        x2_1, x2_2 = self.down1(x1_1, x1_2)
        x3_1, x3_2 = self.down2(x2_1, x2_2)
        x4_1, x4_2 = self.down3(x3_1, x3_2)
        x5_1, x5_2 = self.down4(x4_1, x4_2)

        out_l, out_r = self.up1(x5_1, x5_2, x4_1, x4_2)
        out_l, out_r = self.up2(out_l, out_r, x3_1, x3_2)
        out_l, out_r = self.up3(out_l, out_r, x2_1, x2_2)
        out_l, out_r = self.up4(out_l, out_r, x1_1, x1_2)
        out_l, out_r = self.outc(out_l, out_r)
        return out_l, out_r
