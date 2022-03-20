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


def get_celeba_transforms(config, split):
    """Transforms taken from https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py"""
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if split == "train" and config.dataset.data_augmentation:
        transform = transforms.Compose(
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
        transform = transforms.Compose(
            [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize(config.dataset.target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    return transform


class CelebA(Dataset):
    """Wrapper around CelebA dataset for implementing additional functionality"""

    def __init__(self, config, split: str = 'train'):
        super().__init__()
        self.root = os.path.join(config.dataset.root, "celeba")
        self.split = split
        self.transform = get_celeba_transforms(config, split)

        # [1] Load attributes for specified split
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
        self.attr = torch.where(self.attr > 0, torch.ones(self.attr.shape), torch.zeros(self.attr.shape)).to(torch.long)
        self.attr_names = list(attr.columns)  # for task labels x subgroup labels

        # [2] Load group attributes and labels
        assert len(config.dataset.groupings) > 0, "Empty grouping list passed in."
        self.subgroup_labels = config.dataset.subgroup_labels
        self.grouping = get_grouping_object(config.dataset.groupings)
        self.task_labels = self.grouping.task_labels
        self.subgroup_attributes = self.grouping.subgroup_attributes
        self.task_label_indices = np.array([self.attr_names.index(tl) for tl in self.task_labels])

        # [3] Create class/group splits (self.counts = group counts, self.classes = class counts)
        self.subgroup_combinations = {}
        self.task_comb_indices = {}
        self.subgroups = []
        if config.dataset.subgroup_labels:
            self.partition_g()
        self.partition_y()

        # [4] Apply at most 1 data balancing method
        assert not (config.dataset.subsampler is not None and config.dataloader.sampler is not None), \
            f"Should only specify at most one of dataset.subsampler or dataloader.sampler"

        if split == "train":
            if config.dataset.subsampler is not None:
                self.subsample(config)
            elif config.dataloader.sampler is not None:
                self.reweight(config)

    def partition_y(self):
        task_labels = self.attr[:, self.task_label_indices]
        pos_classes = torch.sum(task_labels, dim=0)
        neg_classes = self.attr.shape[0] - pos_classes

        self.classes = torch.zeros((len(self.task_label_indices), 2)).type(torch.LongTensor)
        self.classes[:, 0] = neg_classes
        self.classes[:, 1] = pos_classes
        logger.info(f"Class counts: {self.classes.tolist()}")

    def partition_g(self):
        """Partitions the examples in this dataset split into groups based on config.dataset.groupings"""
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

        logger.info(f"Split                : {self.split}")
        logger.info(f"Subgroup attributes  : {self.subgroup_attributes}")
        logger.info(f"Subgroup combinations: {self.subgroup_combinations}")

        bin_counts = []
        for channel in range(self.subgroups.shape[1]):
            bin_counts.append(torch.bincount(self.subgroups[:, channel]))
        self.counts = torch.stack(bin_counts, dim=0)
        logger.info(f'Subgroup counts: {self.counts.tolist()}')

    def reweight(self, config):
        """Computes class/group weights for dataloader sampling"""
        if len(self.task_labels) > 1:
            logger.info("Reweighting only implemented for single task, but multiple are detected. Setting all weights to 1.")
            if config.dataloader.sampler == "rwg":
                self.rw_labels = [1.0] * len(self.subgroups)
            elif config.dataloader.sampler == "rwy":
                self.rw_labels = [1.0] * len(self.attr)
            else:
                raise ValueError(f"Unrecognized sampler value: {config.dataloader.sampler}")
            return

        if config.dataloader.sampler == "rwg":
            self.rw_labels = [float(len(self)) / size.float().item() for size in self.counts[0]]  # 0-index for STL
        elif config.dataloader.sampler == "rwy":
            self.rw_labels = [float(len(self)) / size.float().item() for size in self.classes[0]]  # 0-index for STL
        else:
            raise ValueError(f"Unrecognized sampler value: {config.dataloader.sampler}")
        logger.info("Reweight sampler (%s): %s", config.dataloader.sampler, self.rw_labels)

    def subsample(self, config):
        """Subsamples the dataset into equally sized classes/groups"""

        if len(self.task_labels) > 1:
            logger.info("Subsampling only implemented for single task, but multiple are detected. Skipping subsampling.")
            return

        perm = torch.randperm(len(self)).tolist()

        if config.dataset.subsampler == "subg":
            min_size = torch.min(self.counts).item()
            counts_g = [0] * self.counts.shape[1]
        elif config.dataset.subsampler == "suby":
            min_size = torch.min(self.classes).item()
            counts_y = [0] * (len(self.task_label_indices) + 1)
        else:
            raise ValueError(f"Unrecognized subsampler value: {config.dataset.subsampler}")
        logger.info("Subsampler (%s) min size: %s", config.dataset.subsampler, min_size)

        sub_indices = []
        for p in perm:
            if config.dataset.subsampler == "subg":
                g = self.subgroups[p, 0].item()
                if counts_g[g] < min_size:
                    counts_g[g] += 1
                    sub_indices.append(p)

            elif config.dataset.subsampler == "suby":
                y = self.attr[p, self.task_label_indices].item()
                if counts_y[y] < min_size:
                    counts_y[y] += 1
                    sub_indices.append(p)

        self.attr = self.attr[sub_indices, :]
        self.filename = self.filename[sub_indices]

        if self.subgroup_labels:
            self.subgroups = self.subgroups[sub_indices, :]

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))
        image = self.transform(image)
        label = self.attr[index, self.task_label_indices]

        # Return type: [image index, image, image label, image subgroup, image weight]
        if self.subgroup_labels:
            subgroup_label = self.subgroups[index]
            return index, image, label.to(image.dtype), subgroup_label, np.float32(1.0)
        else:
            return index, image, label.to(image.dtype), 0, np.float32(1.0)  # dummy group (everything is group 0)

    def __len__(self):
        return len(self.attr)
