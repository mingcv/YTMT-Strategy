import os
from os.path import join

import torch
from PIL import Image

import data.torchdata as torchdata
from data.transforms import to_tensor


class MoireDataset(torchdata.Dataset):
    def __init__(self, datadir, fns=None, size=None, phase='train'):
        super(MoireDataset, self).__init__()
        self.size = size
        self.datadir = datadir
        self.dirs = ['moire', 'clear'] if phase == 'train' else ['ValidationMoire', 'ValidationClear']
        self.fns = fns or os.listdir(join(datadir, self.dirs[0]))
        if size is not None:
            self.fns = self.fns[:size]

    def __getitem__(self, index):
        fn = self.fns[index]
        moire_img = Image.open(join(self.datadir, self.dirs[0], fn)).convert('RGB')
        clear_img = Image.open(join(self.datadir, self.dirs[1], fn)).convert('RGB')
        moire_img = moire_img.resize((320, 320))
        clear_img = clear_img.resize((320, 320))
        moire_img = to_tensor(moire_img)
        clear_img = to_tensor(clear_img)

        data = {'input': moire_img, 'target_t': clear_img,
                'target_r': torch.clip(clear_img - moire_img, 0, 1),
                'fn': fn}
        return data

    def __len__(self):
        if self.size is not None:
            return min(len(self.fns), self.size)
        else:
            return len(self.fns)
