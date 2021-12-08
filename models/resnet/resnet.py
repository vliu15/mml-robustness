import logging

import numpy as np
import torch
import torch.nn.functional as F

from datasets.groupings import get_grouping_object
from models.base import ClassificationModel
from models.resnet.modules import Bottleneck, ResNet as _ResNet

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class ResNet(ClassificationModel):
    """Wrapper around ResNet in case of extra functionalities that need to be implemented"""

    def __init__(self, config, block, layers):
        super().__init__()
        self.config = config

        # TODO: this might be hacky since the grouping is also instantiated in celeba.py
        self.grouping = get_grouping_object(config.dataset.groupings)

        self.resnet = _ResNet(
            block=block,
            layers=layers,
            num_classes=len(config.dataset.groupings),
            zero_init_residual=config.model.zero_init_residual,
            groups=config.model.groups,
            width_per_group=config.model.width_per_group,
            replace_stride_with_dilation=config.model.replace_stride_with_dilation,
            norm_layer=config.model.norm_layer,
        )

        if config.model.pretrained:
            from torchvision.models.utils import load_state_dict_from_url
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth", progress=True)
            # NOTE(vliu15): throw out the fc weights since these are linear projections to ImageNet classes
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            self.resnet.load_state_dict(state_dict, strict=False)
            logging.info("Downloaded and loaded pretrained ResNet-50")

    def predict(self, x):
        return self.resnet(x)

    ### can only support binary in this setting due to no seperate linear layer per task
    def forward(self, x, y, w, first_batch_loss=None):
        # Forward pass
        logits = self.resnet(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

        # Apply loss weighting
        loss = loss * w.reshape(-1, 1)

        # Apply multi task weighting to loss
        task_weights = torch.tensor(self.config.dataset.task_weights, device=loss.device).float()
        loss = loss * task_weights.unsqueeze(dim=0)

        ## loop over columns in logits
        output_dict = {}
        for col, name in zip(range(logits.shape[1]), self.grouping.task_labels):
            task_logits = logits[:, col]
            y_task = y[:, col]
            task_loss = loss[:, col].mean()
            output_dict[f"loss_{name}"] = task_loss

            with torch.no_grad():
                accuracy = ((task_logits > 0.0) == y_task.bool()).float().mean()
                ## in the case of jtt only one task and hence:
                output_dict[f"metric_{name}_avg_acc"] = accuracy

        ### loss based task weighting
        if self.config.dataset.get("loss_based_task_weighting", False):
            loss_batch_mean = loss.mean(dim=0)

            if first_batch_loss is None:
                output_dict['first_batch_loss'] = loss_batch_mean
                output_dict["loss"] = torch.sum(
                    loss_batch_mean * torch.pow(torch.ones(logits.shape[1], device=device), self.config.dataset.lbtw_alpha)
                )
            else:
                new_task_weights = []
                for col in range(logits.shape[1]):
                    new_task_weights.append(loss_batch_mean[col] / first_batch_loss[col])

                new_task_weights = torch.tensor(new_task_weights, device=device).float()
                output_dict["loss"] = torch.sum(loss_batch_mean * torch.pow(new_task_weights, self.config.dataset.lbtw_alpha))

        else:
            ## sum loss on channel then take mean
            loss = loss.sum(dim=1)
            output_dict["loss"] = loss.mean()  # NOTE(vliu15) all tasks are weighted equally here

        output_dict["yh"] = logits.detach()
        return output_dict

    ### can only support binary in this setting due to no seperate linear layer per task
    def forward_subgroup(self, x, y, g, w, first_batch_loss=None):
        logits = self.resnet(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

        # Apply loss weighting
        loss = loss * w.reshape(-1, 1)

        ## apply multi task weighting to loss
        task_weights = torch.tensor(self.config.dataset.task_weights, device=loss.device, dtype=loss.dtype)
        loss = loss * task_weights.unsqueeze(dim=0)

        output_dict = {}
        with torch.no_grad():

            # [1] Loop over tasks first
            for i, task in enumerate(self.grouping.task_labels):
                task_logits = logits[:, i]
                y_task = y[:, i]
                g_task = g[:, i]
                task_loss = loss[:, i]

                # [1.1] Compute task average
                output_dict[f"loss_{task}"] = task_loss.mean()
                output_dict[f"metric_{task}_avg_acc"] = ((task_logits > 0.0) == y_task.bool()).float().mean()

                # Only run this in eval since we don't log batch metrics in training
                # since not all subgroups are guaranteed to be present
                if not self.training:

                    # [1.2] Compute subgroup averages
                    for j in range(2**(len(self.grouping.subgroup_attributes[task]) + 1)):
                        logits_subgroup = task_logits[(g_task == j).nonzero(as_tuple=True)[0]]
                        y_subgroup = y_task[(g_task == j).nonzero(as_tuple=True)[0]]

                        # Store accuracy components as counts in the format [correct, total] in numpy arrays so they can be easily added
                        subgroup_counts_key = f"metric_{task}_g{j}_counts"
                        if logits_subgroup.shape[0] == 0:
                            output_dict[subgroup_counts_key] = np.array([0, 0], dtype=np.float32)
                        else:
                            num_correct = ((logits_subgroup > 0.0) == y_subgroup.bool()).float().sum().item()
                            output_dict[subgroup_counts_key] = np.array(
                                [num_correct, logits_subgroup.shape[0]], dtype=np.float32
                            )

        ### loss based task weighting
        if self.config.dataset.get("loss_based_task_weighting", False):
            loss_batch_mean = loss.mean(dim=0)
            if first_batch_loss is None:
                output_dict['first_batch_loss'] = loss_batch_mean
                output_dict["loss"] = torch.sum(
                    loss_batch_mean *
                    torch.pow(torch.ones(logits.shape[1], device=loss.device), self.config.dataset.lbtw_alpha)
                )
            else:
                new_task_weights = []
                for col in range(logits.shape[1]):
                    new_task_weights.append(loss_batch_mean[col] / first_batch_loss[col])

                new_task_weights = torch.tensor(new_task_weights, device=loss.device, dtype=loss.dtype)
                output_dict["loss"] = torch.sum(loss_batch_mean * torch.pow(new_task_weights, self.config.dataset.lbtw_alpha))

        else:
            ## sum loss on channel then take mean
            loss = loss.sum(dim=1)
            output_dict["loss"] = loss.mean()

        output_dict["yh"] = logits.detach()
        return output_dict


class ResNet50(ResNet):
    """Wrapper around ResNet50"""

    def __init__(self, config):
        super().__init__(config=config, block=Bottleneck, layers=[3, 4, 6, 3])
