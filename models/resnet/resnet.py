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

        if config.model.pretrained:
            from torchvision.models.utils import load_state_dict_from_url
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-0676ba61.pth", progress=True)
            # NOTE(vliu15): throw out the fc weights since these are linear projections to ImageNet classes
            state_dict.pop("fc.weight")
            state_dict.pop("fc.bias")
            self.resnet.load_state_dict(state_dict, strict=False)
            print("Downloaded and loaded pretrained ResNet-50")

    def forward(self, x, y):
        # Forward pass
        logits = self.resnet(x)

        # BCE loss and accuracy
        loss = F.binary_cross_entropy_with_logits(logits, y)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            accuracy = ((probs > 0.5) == y.bool()).float().mean()

        return {
            "loss": loss,
            "metric_acc": accuracy,
            "yh": probs,
        }


class ResNet50(ResNet):
    """Wrapper around ResNet50"""

    def __init__(self, config):
        super().__init__(config=config, block=Bottleneck, layers=[3, 4, 6, 3])
