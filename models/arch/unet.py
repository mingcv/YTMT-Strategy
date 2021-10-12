import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CoReLUAttBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, fusion=True, fusion_rate=2, dilation=1):
        super(CoReLUAttBlock, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        if fusion:
            self.fusion = nn.Conv2d(in_channels * fusion_rate, in_channels, kernel_size=1, dilation=dilation)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU()

    def forward(self, input_features, additional_features=None):
        if additional_features is not None:
            input_features = torch.cat([input_features, additional_features], dim=1)
            input_features = self.fusion(input_features)
        input_features = self.model(input_features)
        if self.norm:
            input_features = self.norm(input_features)
        out_features = self.relu(input_features)
        return out_features, input_features - out_features


class Down(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, fusion=True, fusion_rate=1, dilation=1):
        super(Down, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        if fusion:
            self.fusion = nn.Conv2d(in_channels * fusion_rate, in_channels, kernel_size=1, dilation=dilation)

        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU()

    def forward(self, input_features, additional_features=None):
        if additional_features is not None:
            input_features = torch.cat([input_features, additional_features], dim=1)
            input_features = self.fusion(input_features)
        input_features = self.model(input_features)
        if self.norm:
            input_features = self.norm(input_features)
        out_features = self.relu(input_features)
        return out_features, input_features - out_features


class Up(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, padding=None, stride=1,
                 norm=False, bias=True, fusion=True, fusion_rate=1, dilation=1):
        super(Up, self).__init__()
        padding = padding or dilation * (kernel_size - 1) // 2
        if fusion:
            self.fusion_1 = nn.Conv2d(in_channels * fusion_rate, in_channels, kernel_size=1, dilation=dilation)
            self.fusion_2 = nn.Conv2d(in_channels * fusion_rate, in_channels, kernel_size=1, dilation=dilation)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride, bias=bias, dilation=dilation),
            PALayer(out_channels),
            CALayer(out_channels)
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.relu = nn.ReLU()

    def forward(self, x1_1, x1_2, x2_1, x2_2):
        x1 = self.fusion_1(torch.cat([x1_1, x1_2], dim=1))
        x2 = self.fusion_2(torch.cat([x2_1, x2_2], dim=1))
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.model(x)
        if self.norm:
            x = self.norm(x)
        out_features = self.relu(x)
        return out_features, x - out_features


class CoOutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True, act=True, fusion=True,
                 fusion_rate=1, dilation=1):
        super().__init__()
        self.fusion = nn.Conv2d(in_channels * fusion_rate, in_channels, kernel_size=1) if fusion else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias,
                              dilation=dilation)
        self.act = nn.Sigmoid() if act else None

    def forward(self, x1, x2=None, residual=None):
        if x2 is not None:
            out = torch.cat([x1, x2], dim=1)
            if residual is not None and residual.shape == out.shape:
                out += residual
            out = self.fusion(out)
        else:
            out = x1
        out = self.conv(out)
        if self.act:
            out = self.act(out)

        return out


class YTMTUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, act=True):
        super(YTMTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_1 = CoReLUAttBlock(n_channels, 64)
        self.inc_2 = CoReLUAttBlock(n_channels, 64)
        self.down1_1 = Down(64, 128, fusion_rate=2)
        self.down1_2 = Down(64, 128, fusion_rate=2)
        self.down2_1 = Down(128, 256, fusion_rate=2)
        self.down2_2 = Down(128, 256, fusion_rate=2)
        self.down3_1 = Down(256, 512, fusion_rate=2)
        self.down3_2 = Down(256, 512, fusion_rate=2)
        factor = 2
        self.down4_1 = Down(512, 1024 // factor, fusion_rate=2)
        self.down4_2 = Down(512, 1024 // factor, fusion_rate=2)
        self.up1_1 = Up(1024, 512 // factor, fusion_rate=1)
        self.up1_2 = Up(1024, 512 // factor, fusion_rate=1)
        self.up2_1 = Up(512, 256 // factor, fusion_rate=1)
        self.up2_2 = Up(512, 256 // factor, fusion_rate=1)
        self.up3_1 = Up(256, 128 // factor, fusion_rate=1)
        self.up3_2 = Up(256, 128 // factor, fusion_rate=1)
        self.up4_1 = Up(128, 64, fusion_rate=1)
        self.up4_2 = Up(128, 64, fusion_rate=1)
        self.outc_1 = CoOutConv(64 * 2, n_classes, act=act)
        self.outc_2 = CoOutConv(64 * 2, n_classes, act=act)

    def forward(self, x, y=None, fn=None):
        x1_1_1, x1_1_2 = self.inc_1(x)
        x1_2_1, x1_2_2 = self.inc_2(y if y is not None else x)
        x2_1_1, x2_1_2 = self.down1_1(x1_1_1, x1_2_2)
        x2_2_1, x2_2_2 = self.down1_2(x1_2_1, x1_1_2)

        x3_1_1, x3_1_2 = self.down2_1(x2_1_1, x2_2_2)
        x3_2_1, x3_2_2 = self.down2_2(x2_2_1, x2_1_2)

        x4_1_1, x4_1_2 = self.down3_1(x3_1_1, x3_2_2)
        x4_2_1, x4_2_2 = self.down3_2(x3_2_1, x3_1_2)

        x5_1_1, x5_1_2 = self.down4_1(x4_1_1, x4_2_2)
        x5_2_1, x5_2_2 = self.down4_2(x4_2_1, x4_1_2)

        out_l_1, out_l_2 = self.up1_1(x5_1_1, x5_2_2, x4_1_1, x4_2_2)
        out_r_1, out_r_2 = self.up1_2(x5_2_1, x5_1_2, x4_2_1, x4_1_2)

        (out_l_1, out_l_2), (out_r_1, out_r_2) = self.up2_1(out_l_1, out_r_2, x3_1_1, x3_2_2), \
                                                 self.up2_2(out_r_1, out_l_2, x3_2_1, x3_1_2)
        (out_l_1, out_l_2), (out_r_1, out_r_2) = self.up3_1(out_l_1, out_r_2, x2_1_1, x2_2_2), \
                                                 self.up3_2(out_r_1, out_l_2, x2_2_1, x2_1_2)

        (out_l_1, out_l_2), (out_r_1, out_r_2) = self.up4_1(out_l_1, out_r_2, x1_1_1, x1_2_2), \
                                                 self.up4_2(out_r_1, out_l_2, x1_2_1, x1_1_2)

        out_l, out_r = self.outc_1(out_l_1, out_r_2), self.outc_2(out_r_1, out_l_2)
        return out_l, out_r



if __name__ == '__main__':
    x = torch.rand(1, 1465, 224, 224)
    net = YTMTUNet(1465, 3)
    print(net)
    print(net(x))
