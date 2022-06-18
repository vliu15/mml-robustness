import logging
import logging.config

from datasets.groupings import get_grouping_object
from models.base import ClassificationModel
from models.clip.modules import CLIPResNet as _CLIPResNet
from models.clip.utils import download_clip, load_resnet
from models.resnet.resnet import ResNet

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class CLIPResNetConfig(object):
    embed_dim: int = 1024  # this is checked below to make sure that it matches the checkpoint embed_dim
    width: int = 64


class CLIPResNet(ResNet):
    """Wrapper around ResNet in case of extra functionalities that need to be implemented"""

    def __init__(self, config, layers):
        # Init as ClassificationModel to avoid downloading/creating unused ResNet model
        super(ClassificationModel, self).__init__()
        self.config = config

        # TODO: this might be hacky since the grouping is also instantiated in celeba.py
        self.grouping = get_grouping_object(config.dataset.groupings)

        self.resnet = _CLIPResNet(
            layers=layers,
            num_classes=len(config.dataset.groupings),
            input_resolution=config.dataset.target_resolution[0],  # could be that (w, h) or indices (0, 1) are different
            output_dim=CLIPResNetConfig.embed_dim,
            heads=CLIPResNetConfig.width * 32 // 64,  # as in https://github.com/openai/CLIP/blob/main/clip/model.py#L264,
            width=CLIPResNetConfig.width,
        )

        if config.model.pretrained:
            # NOTE(vliu15): download link is currently hard-coded to be ResNet-50 weights
            pt = download_clip(
                "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
            )
            state_dict, embed_dim = load_resnet(pt)
            assert CLIPResNetConfig.embed_dim == embed_dim, f"Default embed_dim={CLIPResNetConfig.embed_dim} doesn't match checkpoint embed_dim={embed_dim}"
            self.resnet.load_state_dict(state_dict, strict=False)
            logging.info("Downloaded and loaded pretrained CLIP ResNet-50")


class CLIPResNet50(CLIPResNet):

    def __init__(self, config):
        super().__init__(config=config, layers=[3, 4, 6, 3])
