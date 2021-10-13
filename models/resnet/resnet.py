import torch
import torch.nn.functional as F

from models.base import ClassificationModel
from models.resnet.modules import Bottleneck, ResNet as _ResNet


class ResNet(ClassificationModel):
    """Wrapper around ResNet in case of extra functionalities that need to be implemented"""

    def __init__(self, config, block, layers):
        super().__init__()
        self.resnet = _ResNet(
            block=block,
            layers=layers,
            num_classes=config.model.num_classes,
            zero_init_residual=config.model.zero_init_residual,
            groups=config.model.groups,
            width_per_group=config.model.width_per_group,
            replace_stride_with_dilation=config.model.replace_stride_with_dilation,
            norm_layer=config.model.norm_layer,
        )
        self.is_multiclass = config.model.is_multiclass

    def forward(self, x, y):
        logits = self.resnet(x)
        if self.is_multiclass:
            loss = F.cross_entropy(logits, y)
            with torch.no_grad():
                accuracy = (logits.argmax(-1) == y).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y)
            with torch.no_grad():
                accuracy = ((logits > 0.5) == y.bool()).float().mean()

        return {
            "loss": loss,
            "metric_acc": accuracy,
        }


class ResNet50(ResNet):
    """Wrapper around ResNet50"""

    def __init__(self, config):
        super().__init__(config=config, block=Bottleneck, layers=[3, 4, 6, 3])
