import os

import pandas
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import itertools


class CelebA(Dataset):
    """Wrapper around CelebA dataset for implementing additional functionality"""

    def __init__(self, config, train: bool = True):
        super().__init__()
        self.root = os.path.join(config.dataset.root, "celeba")

        self.task_labels = config.dataset.task_labels 
        self.subgroup_labels = config.dataset.subgroup_labels
        self.subgroup_attributes = config.dataset.subgroup_attributes
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
                comb_group_label =  {combinations[i]:i for i in range(len(combinations))}
                self.subgroup_combinations[key] = comb_group_label
                
            
            for ind_attrr in self.attr:
                
                group_label = []
                for key in self.task_comb_indices.keys():
                    indices = self.task_comb_indices[key]
                    tup_to_group_label = tuple(ind_attrr[indices].tolist())
                    group_label.append(self.subgroup_combinations[key][tup_to_group_label])

                self.subgroups.append(group_label)
    

            self.subgroups = torch.tensor(self.subgroups)

            if train:
                print(self.subgroup_attributes)
                print(self.subgroup_combinations)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, "img_align_celeba", self.filename[index]))
        image = self.transform(image)
        label = self.attr[index, self.task_label_indices]
        if self.subgroup_labels:
            subgroup_label = self.subgroups[index]
            return image, label.to(image.dtype), subgroup_label.to(image.dtype)
        else:
            return image, label.to(image.dtype)

    def __len__(self):
        return len(self.attr)
