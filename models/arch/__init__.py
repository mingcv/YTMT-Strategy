import os

from models.arch.ytmt_networks import *
from models.arch.unet import *
from models.arch.dncnn import *


def ytmt_pas(in_channels, out_channels, **kwargs):
    return YTMT_PS(in_channels, out_channels, pattern='A', **kwargs)


def ytmt_pcs(in_channels, out_channels, **kwargs):
    return YTMT_PS(in_channels, out_channels, pattern='C', **kwargs)


def ytmt_uas(in_channels, out_channels, **kwargs):
    return YTMT_US(in_channels, out_channels, pattern='A', **kwargs)


def ytmt_ucs_old(in_channels, out_channels, **kwargs):
    return YTMTUNet(in_channels, out_channels, **kwargs)


def ytmt_ucs(in_channels, out_channels, **kwargs):
    return YTMT_US(in_channels, out_channels, pattern='C', **kwargs)


if __name__ == '__main__':
    x = torch.ones(1, 1475, 256, 256)
    net = ytmt_ucs(1475, 3)
    print(net)
    url = "./tmp.pth"
    torch.save(net.state_dict(), url)
    print('\n', os.path.getsize(url) / (1024 * 1024), 'MB')
    l, r = net(x)
