import os
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FilteredDataset(Dataset):

    def __init__(self, config: OmegaConf, dataset: Dataset, filters: Dict[str, int] = {}):
        super().__init__()

        self.preprocess_transforms = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(config.dataset.target_resolution),
                transforms.ToTensor(),
            ]
        )

        config.dataset.groupings = []  # so that we are able to grab all attributes
        self.dataset = dataset

        attr = self.dataset.attr.numpy()

        instances = set(range(len(self.dataset)))
        for label, value in filters.items():
            label = self.dataset.attr_names.index(label)
            filtered_instances = np.argwhere(attr[:, label] == value).squeeze(-1).tolist()
            instances.intersection_update(filtered_instances)

        self.instances = list(sorted(instances))
        print(f"There are {len(self.instances)} examples that satisfy the given filters:")
        for label, value in filters.items():
            print(f"\t{label}:{value}")

    def __getitem__(self, index):
        if len(self.instances) == 0:
            return 1, torch.zeros((3, 224, 224), dtype=torch.float32), -1 * torch.ones((1,), dtype=torch.int32), 1, 1
        return self.dataset.__getitem__(self.instances[index])

    def get_original_image(self, index):
        if len(self.instances) == 0:
            return torch.zeros((3, 224, 224), dtype=torch.uint8)
        image = Image.open(os.path.join(self.dataset.root, "img_align_celeba", self.dataset.filename[self.instances[index]]))
        image = image.convert("RGB")
        image = self.preprocess_transforms(image)
        return (image * 255).to(torch.uint8)

    def __len__(self):
        return len(self.instances)
