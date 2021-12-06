import logging
import pickle

from torch.utils.data import Dataset

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class UpsampledDataset(Dataset):
    """
    Wrapper around the original training dataset to upsample specified examples more often than the others

    This works under the hood by mapping all the indices above the range of the initial dataset into
    indices of the examples that should be upsampled

    Args:
        dataset: original dataset implementation
        upsample_pkl: path to pickle of indices of examples in the original dataset that are already upsampled
    """

    def __init__(self, dataset: Dataset, upsample_pkl: str):
        super().__init__()
        with open(upsample_pkl, "rb") as f:
            data = pickle.load(f)
            upsample_indices = data["error_set"]
            pickle_metas = data["meta"]

        for task, pickle_meta in pickle_metas.items():
            logger.info(f"Meta for {task}")
            for subgroup, (errors, total) in pickle_meta.items():
                logger.info(f"  {subgroup}: {errors} errors / {total} total || {100 * errors / total:.4f}%")

        self.dataset = dataset
        self.upsample_indices = upsample_indices

    def __getitem__(self, index):
        # Return original dataset if index is in range
        if index < len(self.dataset):
            return self.dataset.__getitem__(index)

        # Otherwise shift to start at 0 and find index
        return self.dataset.__getitem__(self.upsample_indices[index - len(self.dataset)])

    def __len__(self):
        return len(self.dataset) + len(self.upsample_indices)
