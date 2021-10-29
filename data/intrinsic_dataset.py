import math
import os.path
from os.path import join

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, AnyStr, Any
from .image_folder import is_image_file, make_dataset
from PIL import Image
import numpy as np

from .transforms import to_tensor


def linear_transform(x, alpha: float = (math.e - 1), beta: float = 1):
    """
    Mapping x to alpha * x + beta, which transforms [l,r] to [alpha * l + beta, alpha * r + beta]
    """
    return alpha * x + beta


class IntrinsicDataset(Dataset):
    def __init__(self):
        super(IntrinsicDataset, self).__init__()

    @staticmethod
    def __get_transform__(data: dict):
        for key in data.keys():
            data[key] = linear_transform(data[key])
        return data


class MITIntrinsicDataset(IntrinsicDataset):
    """
        Dataset origin: https://www.cs.toronto.edu/~rgrosse/intrinsic/

        Following the train-test split strategy and dataset pre-processing script in
        "Direct intrinsics: Learning albedo-shading decomposition by convolutional regression." CVPR 2015.
        The dataset pre-processing script can be found in
        https://github.com/tnarihi/direct-intrinsics/tree/master/data/mit.

    """
    objects = {
        "train": ['apple', 'box', 'cup1', 'dinosaur', 'frog1', 'panther', 'paper1', 'phone', 'squirrel', 'teabag2'],
        "test": ['cup2', 'deer', 'frog2', 'paper2', 'pear', 'potato', 'raccoon', 'sun', 'teabag1', 'turtle']}

    @staticmethod
    def no_transform(item):
        return item

    def __init__(
            self,
            root: str,
            size: int = None,
            transforms: str = None,
            train: bool = True
    ) -> None:
        super(MITIntrinsicDataset, self).__init__()
        self.size = size
        self.root = root.rstrip('/').rstrip('\\')
        self.train = "train" if train else "test"
        self.transforms = transforms
        self.objs = self.objects[self.train]
        self.dirs = {k: dict() for k in self.objs}
        for key in self.dirs.keys():
            self.dirs[key]["mask"] = self.root + '/' + key + '/' + 'mask.png'
            self.dirs[key]["albedo"] = self.root + '/' + key + '/' + 'reflectance.png'
            self.dirs[key]["shading"] = [self.root + '/' + key + '/' + ('gray_shading%02d.png' % idx) for idx in
                                         range(1, 11)]
            self.dirs[key]["shading"].append(self.root + '/' + key + '/' + 'gray_shading.png')

            self.dirs[key]["input"] = [self.root + '/' + key + '/' + ('light%02d.png' % idx) for idx in range(1, 11)]
            self.dirs[key]["input"].append(self.root + '/' + key + '/' + "diffuse.png")
        # check whether images exists or not
        if self.size is None:
            self.size = 110
        for k, v in self.dirs.items():
            for ptp, pth in v.items():
                if type(pth) == list:
                    for inner_pth in pth:
                        if not os.path.isfile(inner_pth):
                            raise FileNotFoundError(inner_pth)
                else:
                    if not os.path.isfile(pth):
                        raise FileNotFoundError(pth)

    def __getpath__(self,
                    index: int
                    ) -> Dict[AnyStr, Any]:

        if index > self.size:
            raise IndexError("out of the range")

        selected_obj = self.objs[index // 11]

        return {"mask": self.dirs[selected_obj]["mask"],
                "albedo": self.dirs[selected_obj]["albedo"],
                "shading": self.dirs[selected_obj]["shading"][index % 11],
                "input": self.dirs[selected_obj]["input"][index % 11]}

    def __getitem__(
            self,
            index: int
    ) -> Dict[AnyStr, Any]:
        paths = self.__getpath__(index)
        ret = dict()
        img_mask = Image.open(paths["mask"]).convert('L')
        img_albedo = Image.open(paths["albedo"]).convert("RGB")
        img_shading = Image.open(paths["shading"]).convert("RGB")
        img_original_shading = Image.open(paths["shading"]).convert("L")
        img_input = Image.open(paths["input"]).convert("RGB")

        ret["mask"] = to_tensor(img_mask)
        ret["mask"][ret["mask"] < 0.5] = 0
        ret["mask"][ret["mask"] > 0.5] = 1
        ret["albedo"] = to_tensor(img_albedo) * ret["mask"]
        ret["shading"] = to_tensor(img_shading) * ret["mask"]

        ret["input"] = linear_transform(to_tensor(img_input) * ret["mask"])
        ret["org_input"] = to_tensor(img_input) * ret["mask"]
        ret["org_shading"] = to_tensor(img_original_shading) * ret["mask"]
        ret["fn"] = "".join((paths["input"]).strip("/").split("/")[-2:])
        # print(ret["fn"])
        return ret
        # return self.__get_transform__(ret)

    def __len__(self):
        return self.size


class BatchInferenceDataset(IntrinsicDataset):
    """
    This inference-only dataset class loads images from the given path.
    """

    def __init__(
            self,
            root: str,
            size: int = None,
            transforms: str = None,
    ) -> None:

        super(BatchInferenceDataset, self).__init__()
        self.size = size
        self.root = root.rstrip('/').rstrip('\\')
        self.transforms = transforms
        self.images = make_dataset(root)
        print(self.images)
        # check whether images exists or not
        if self.size is None:
            self.size = len(self.images)

    def __getitem__(
            self,
            index: int
    ) -> Dict[AnyStr, Any]:
        if index > self.size:
            raise IndexError("out of the range")
        ret = dict()
        ret["fn"] = "%d" % index
        path = self.images[index]
        img = Image.open(path)
        ret["org_input"] = to_tensor(img)
        ret["mask"] = torch.ones(ret["org_input"].shape).type(torch.FloatTensor)
        ret["input"] = linear_transform(ret["org_input"])
        print(ret)
        return ret

    def __len__(self):
        return self.size
