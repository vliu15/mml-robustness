"""Contains the entire train function for both train.py and train_ddp.py."""

import logging
import os
import pickle

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.base import ClassificationModel
from train_common import to_device, train
from utils.init_modules import init_ema, init_logdir, init_model, init_optimizer, init_scheduler

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


class UpsampledDataset(torch.utils.data.Dataset):
    """
    Wrapper around the original training dataset to upweight specified examples more often than the others

    This works under the hood by mapping all the indices above the range of the initial dataset into
    indices of the examples that should be upweighted

    Args:
        dataset: original dataset implementation
        lambda_up: the upsampling factor for specified examples
        upsample_indices: indices of examples in the original dataset that should be upsampled
    """

    def __init__(self, dataset, lambda_up, upsample_indices=[]):
        super().__init__()
        assert lambda_up > 1 and isinstance(lambda_up, int), \
            f"Upsampling amount should be a positive integer, got lambda_up={lambda_up} instead."

        self.dataset = dataset
        self.lambda_up = lambda_up
        self.upsample_indices = upsample_indices

    def __getitem__(self, index):
        # Return original dataset if index is in range
        if index < len(self.dataset) or self.lambda_up == 1:
            return self.dataset.__getitem__(index)

        # Otherwise shift to start at 0 and take modulo wrt self.upsample_indices
        index -= len(self.dataset)
        return self.dataset.__getitem__(index % len(self.upsample_indices))

    def __len__(self):
        return len(self.dataset) + (self.lambda_up - 1) * len(self.upsample_indices)


@torch.no_grad()
def construct_error_set(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    device: str,
    task: int = 0,
):
    """Constructs the error set of training examples after initial ERM training"""
    model.eval()

    global_indices = []
    postfix = {"error rate": "0 / 0"}

    total_errors, total_examples = 0, 0
    with tqdm(total=len(train_dataloader), desc="Constructing error set", postfix={"error rate": "0/0"}) as pbar:
        for batch in train_dataloader:
            batch = to_device(batch, device)
            output_dict = model.inference_step(batch)

            # NOTE(vliu15): since yh could contain multiple binary predictions per batch example,
            # let's just take the first one to keep things simple and consistent with JTT
            yh = (output_dict["yh"][:, task] > 0).float()
            y = output_dict["y"][:, task]

            errors = (yh != y)
            global_indices += batch[0][errors].cpu().tolist()

            total_errors += errors.sum()
            total_examples += y.numel()

            postfix["error rate"] = f"{total_errors} / {total_examples}"
            pbar.set_postfix(postfix)
            pbar.update(1)

    print(f"Total error rate: {100 * total_errors / total_examples:.4f}%")
    return global_indices


def train_jtt(
    *,
    global_step: int,
    epoch: int,
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: str,
    rank: int = 0,
) -> None:
    """
    Trains the model according to the algorithm proposed in (Liu, Haghgoo, Chen, et al. 2020)
    Just Train Twice: Improving Group Robustness without Training Group Information

    Args: see train() docstring in train_common.py.
    """
    assert isinstance(model, ClassificationModel), \
        f"JTT training is only supported for classification models. Got {model.__class__.__name__} instead."

    ###########
    # Stage 1 #
    ###########

    if hasattr(config.train, "jtt_error_set") and config.train.jtt_error_set:
        with open(os.path.join(config.train.log_dir, "jtt_error_set.pkl"), "rb") as f:
            error_indices = pickle.load(f)
        logger.info(f"Loaded JTT error set. Error rate: {100 * len(error_indices) / len(train_dataloader.dataset):.4f}")

    else:
        # 1. Train f_id on D via ERM for T steps
        train(
            global_step=global_step,
            epoch=epoch,
            config=config,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            writer=writer,
            device=device,
            rank=rank,
        )

        # 2. Construct the error set E of training examples misclassified by f_id
        error_indices = construct_error_set(model, train_dataloader, device, task=0)
        with open(os.path.join(config.train.log_dir, "jtt_error_set.pkl"), "wb") as f:
            pickle.dump(error_indices, f)

    ###########
    # Stage 2 #
    ###########

    # 3. Construct upsampled dataset D_up containing examples in the error set λ_up times and all other examples once
    train_dataset = UpsampledDataset(
        dataset=train_dataloader.dataset,
        lambda_up=config.train.lambda_up,
        upsample_indices=error_indices,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # 4. Train final model f_final on D_up via ERM
    config.train.total_epochs = config.train.stage_2_total_epochs
    config.train.log_dir = os.path.join(config.train.log_dir, "stage_2")  # create new logdir
    init_logdir(config)

    writer = SummaryWriter(config.train.log_dir)
    model = init_model(config).to(device)
    ema = init_ema(config, model)  # re-init model EMA
    optimizer = init_optimizer(config, model)  # re-init optimizer
    scheduler = init_scheduler(config, optimizer)  # re-init scheduler

    train(
        global_step=0,
        epoch=0,
        config=config,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        writer=writer,
        device=device,
        rank=rank,
    )
