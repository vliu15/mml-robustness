import os

import pandas
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CelebA(Dataset):
    """Wrapper around CelebA dataset for implementing additional functionality"""

    def __init__(self, config, train: bool = True):
        super().__init__()
        self.root = os.path.join(config.dataset.root, "celeba")

        # Transforms taken from https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        config.dataset.target_resolution,
                        scale=(0.7, 1.0),
                        ratio=(1.0, 1.3333333333333333),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(orig_min_dim),
                    transforms.Resize(config.dataset.target_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )

        splits = pandas.read_csv(
            os.path.join(self.root, "list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0
        )
        attr = pandas.read_csv(os.path.join(self.root, "list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = (splits[1] == (0 if train else 1))

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))
        image = self.transform(image)
        label = self.attr[index, :]

        # NOTE(vliu15): index 9 corresponds to blonde hair attribute
        label = label[9:10]
        return index, image, label.to(image.dtype)

    def __len__(self):
        return len(self.attr)
