import logging
import pickle

from torch.utils.data import Dataset

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class UpweightedDataset(Dataset):
    """
    Wrapper around the original training dataset to upweight specified examples more often than the others

    This works under the hood by mapping all the indices above the range of the initial dataset into
    indices of the examples that should be upweighted

    Args:
        dataset: original dataset implementation
        lambda_up: the upsampling factor for specified examples
        upweight_pkl: path to pickle of indices of examples in the original dataset that should be upweighted
    """

    def __init__(self, dataset: Dataset, lambda_up: int, upsample_pkl: str):
        super().__init__()
        assert lambda_up > 1 and isinstance(lambda_up, int), \
            f"Upsampling amount should be a positive integer, got lambda_up={lambda_up} instead."

        with open(upsample_pkl, "rb") as f:
            data = pickle.load(f)
            upweight_indices = data["error_set"]
            pickle_meta = data["meta"]

        for key, (errors, total) in pickle_meta.items():
            logger.info(f"{key}: {errors} errors / {total} total || {100 * errors / total:.4f}%")

        self.dataset = dataset
        self.lambda_up = lambda_up
        self.upweight_indices = upweight_indices

    def __getitem__(self, index):
        # Return original dataset if index is in range
        if index < len(self.dataset) or self.lambda_up == 1:
            return self.dataset.__getitem__(index)

        # Otherwise shift to start at 0 and take modulo wrt self.upweight_indices
        up_index = (index - len(self.dataset)) % len(self.upweight_indices)
        up_index = self.upweight_indices[up_index]
        return self.dataset.__getitem__(up_index)

    def __len__(self):
        return len(self.dataset) + (self.lambda_up - 1) * len(self.upweight_indices)
