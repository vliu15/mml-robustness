"""Script for standard ERM training"""

import logging

import hydra
import torch.multiprocessing as multiprocessing

from train_common import train_ddp, train_single
from utils.init_modules import init_logdir

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="default")
def main(config):
    """Entry point into ERM training"""
    config = config.exp
    init_logdir(config)

    max_gpus = torch.cuda.device_count()
    if config.train.n_gpus == -1:
        config.train.n_gpus = max_gpus

    ## make sure task weights are correctly specified
    if len(config.dataset.task_weights) != len(config.dataset.groupings):
        raise ValueError("Task weights must be the same length as task labels")

    if sum(config.dataset.task_weights) != 1:
        if len(set(config.dataset.task_weights)) != 1 or 1 not in config.dataset.task_weights:
            raise ValueError("Ensure task weights are either all 1 (no weighting) or sum up to 1 (weighting)")

    if config.dataset.loss_based_task_weighting:
        if len(set(config.dataset.task_weights)) != 1 or 1 not in config.dataset.task_weights:
            raise ValueError("To apply loss based task weighting, the original task weights must be all 1")

    if config.train.fp16:
        try:
            import torch.cuda.amp
        except ModuleNotFoundError:
            config.train.fp16 = False
            logger.info("torch.cuda.amp.autocast not found, forcing training in FP32")

    # Determine whether to launch single or multi-gpu training
    n_gpus = min(config.train.n_gpus, max_gpus)
    if n_gpus <= 1:
        train_single(config)
    else:
        assert config.train.n_gpus <= max_gpus, f"Specified {config.train.n_gpus} gpus, but only {max_gpus} total"
        multiprocessing.spawn(train_ddp, args=[config.train.n_gpus, config], nprocs=config.train.n_gpus, join=True)


if __name__ == "__main__":
    main()
