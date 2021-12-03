import itertools
import logging
import os

import numpy as np
import pandas
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from datasets.groupings import get_grouping_object

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class CelebA(Dataset):
    """Wrapper around CelebA dataset for implementing additional functionality"""

    def __init__(self, config, split: str = 'train'):
        super().__init__()
        self.root = os.path.join(config.dataset.root, "celeba")

        self.subgroup_labels = config.dataset.subgroup_labels

        self.grouping = get_grouping_object(config.dataset.groupings)
        self.task_labels = self.grouping.task_labels
        self.subgroup_attributes = self.grouping.subgroup_attributes

        # Transforms taken from https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        if split == 'train' and config.dataset.data_augmentation:
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

        if split == 'train':
            marker = 0
        elif split == 'val':
            marker = 1
        else:
            marker = 2
        mask = (splits[1] == marker)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        ## for task labels x subgroup labels

        if len(self.task_labels) == 0:
            self.task_labels = self.attr_names
            self.task_label_indices = np.arange(len(self.attr_names))
        else:
            self.task_label_indices = np.array([self.attr_names.index(tl) for tl in self.task_labels])

        if self.subgroup_labels:
            self.subgroup_combinations = {}
            self.task_comb_indices = {}
            self.subgroups = []

            if len(self.subgroup_attributes.keys()) != len(self.task_labels):
                raise ValueError("Not enough task labels in subgroups attributes")

            for key in self.subgroup_attributes.keys():
                if key not in self.task_labels:
                    raise ValueError("Incorrectly denoted task label")

                cols = [key] + list(self.subgroup_attributes[key])
                self.task_comb_indices[key] = [self.attr_names.index(col) for col in cols]

                subgroup_len = len(self.subgroup_attributes[key])
                combinations = list(itertools.product([0, 1], repeat=subgroup_len + 1))
                comb_group_label = {combinations[i]: i for i in range(len(combinations))}
                self.subgroup_combinations[key] = comb_group_label

            for ind_attr in self.attr:
                group_label = []
                for key in self.task_comb_indices.keys():
                    indices = self.task_comb_indices[key]
                    tup_to_group_label = tuple(ind_attr[indices].tolist())
                    group_label.append(self.subgroup_combinations[key][tup_to_group_label])

                self.subgroups.append(group_label)

            self.subgroups = torch.tensor(self.subgroups, dtype=torch.long)

            if split == 'train':
                logger.info(f"Subgroup attributes  : {self.subgroup_attributes}")
                logger.info(f"Subgroup combinations: {self.subgroup_combinations}")

                ## need to fix this such that it is output per channel
                #torch.bincount(self.subgroups.squeeze(dim=1)).tolist()

                bin_counts = []
                for channel in range(self.subgroups.shape[1]):
                    bin_counts.append(torch.bincount(self.subgroups[:, channel]))
                counts = torch.vstack(bin_counts)
                logger.info(f'Subgroup counts: {counts.detach().cpu().numpy()}')

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))
        image = self.transform(image)
        label = self.attr[index, self.task_label_indices]

        # Return type: [image index, image, image label, image subgroup, image weight]
        if self.subgroup_labels:
            subgroup_label = self.subgroups[index]
            return index, image, label.to(image.dtype), subgroup_label, 1.0
        else:
            return index, image, label.to(image.dtype), 0, 1.0  # dummy group (everything is group 0)

    def __len__(self):
        return len(self.attr)
