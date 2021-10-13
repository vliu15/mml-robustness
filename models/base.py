import torch.nn as nn


class ClassificationModel(nn.Module):
    """Abstract model that handles forward i/o for classification models"""

    def supervised_step(self, batch):
        x, y = batch
        out_dict = self.forward(x, y)
        out_dict["y"] = y
        return out_dict
