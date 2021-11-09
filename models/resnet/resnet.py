import torch
import torch.nn.functional as F

from models.base import ClassificationModel
from models.resnet.modules import Bottleneck, ResNet as _ResNet


class ResNet(ClassificationModel):
    """Wrapper around ResNet in case of extra functionalities that need to be implemented"""

    def __init__(self, config, block, layers):
        super().__init__()
        self.config = config
        self.resnet = _ResNet(
            block=block,
            layers=layers,
            num_classes=config.model.num_tasks,
            zero_init_residual=config.model.zero_init_residual,
            groups=config.model.groups,
            width_per_group=config.model.width_per_group,
            replace_stride_with_dilation=config.model.replace_stride_with_dilation,
            norm_layer=config.model.norm_layer,
        )
        self.is_multiclass = config.model.is_multiclass

        if config.model.pretrained:
            from torchvision.models.utils import load_state_dict_from_url
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth", progress=True)
            # NOTE(vliu15): throw out the fc weights since these are linear projections to ImageNet classes
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            self.resnet.load_state_dict(state_dict, strict=False)
            print("Downloaded and loaded pretrained ResNet-50")

    ### can only support binary in this setting due to no seperate linear layer per task
    def forward(self, x, y):
        logits = self.resnet(x)

        ## loop over columns in logits
        output_dict = {}
        loss = 0

        task_labels = self.config.dataset.task_labels if len(self.config.dataset.task_labels) != 0 else [i for i in range(40)]
        for col, name in zip(range(logits.shape[1]), task_labels):
            task_logits = logits[:, col]
            y_task = y[:, col]
            task_loss = F.binary_cross_entropy_with_logits(task_logits, y_task)
            loss += task_loss
            with torch.no_grad():
                accuracy = ((task_logits > 0.0) == y_task.bool()).float().mean()
                metric_name = f"metric_task_{name}_avg_acc"
                output_dict[metric_name] = accuracy

        output_dict['loss'] = loss
        return output_dict

    ### can only support binary in this setting due to no seperate linear layer per task
    def forward_subgroup(self, x, y, g):
        logits = self.resnet(x)

        logits = self.resnet(x)

        ## loop over columns in logits
        output_dict = {}
        loss = 0

        task_labels = self.config.dataset.task_labels if len(self.config.dataset.task_labels) != 0 else [i for i in range(40)]
        for col, name in zip(range(logits.shape[1]), task_labels):
            task_logits = logits[:, col]
            y_task = y[:, col]
            g_task = g[:, col]
            task_loss = F.binary_cross_entropy_with_logits(task_logits, y_task)
            loss += task_loss
            with torch.no_grad():
                avg_accuracy = ((task_logits > 0.0) == y_task.bool()).float().mean()
                avg_metric_name = f"metric_task_{name}_avg_acc"
                output_dict[avg_metric_name] = avg_accuracy

                for i in range(2**(len(self.config.dataset.subgroup_attributes[name]) + 1)):

                    logits_subgroup = task_logits[(g_task == i).nonzero(as_tuple=True)[0]].cpu()
                    y_subgroup = y_task[(g_task == i).nonzero(as_tuple=True)[0]].cpu()

                    if logits_subgroup.shape[0] == 0:
                        output_dict[subgroup_metric_name] = torch.tensor(0)
                        output_dict[subgroup_count_name] = 0
                    else:
                        subgroup_accuracy = ((logits_subgroup > 0.0) == y_subgroup.bool()).float().mean()
                        subgroup_metric_name = f"metric_task_{name}_subgroup_{i}_acc"
                        subgroup_count_name = f"metric_task_{name}_subgroup_{i}_acc_count"
                        output_dict[subgroup_metric_name] = subgroup_accuracy
                        output_dict[subgroup_count_name] = logits_subgroup.shape[0]

        output_dict['loss'] = loss
        return output_dict


class ResNet50(ResNet):
    """Wrapper around ResNet50"""

    def __init__(self, config):
        super().__init__(config=config, block=Bottleneck, layers=[3, 4, 6, 3])
