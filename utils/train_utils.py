"""Contains training utils for a single rank, abstracted to allow for single or distributed XPU training"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tabulate import tabulate


def to_device(batch, device):
    """Puts a batch onto the specified device"""
    return [b.to(device) for b in batch if isinstance(b, torch.Tensor)]


def seed_all_rng(seed: int, cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def get_human_readable_count(number: int) -> str:
    assert number >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


def get_top_level_summary(model: nn.Module) -> None:
    headers = ["", "Name", "Module", "Params", "Buffers"]
    table = []
    for index, (name, module) in enumerate(model.named_children()):
        params = get_human_readable_count(sum(p.numel() for p in module.parameters() if p.requires_grad))
        buffers = get_human_readable_count(sum(b.numel() for b in module.buffers()))
        table += [[index, name, module.__class__.__name__, params, buffers]]

    model_summary = tabulate(table, headers=headers, tablefmt="pretty", colalign=["left"] * 3 + ["right"] * 2)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    all_params_megabytes = (trainable_params + trainable_params) * 32 / 8 * 1e-6
    all_buffers_megabytes = sum(b.numel() for b in model.buffers()) * 32 / 8 * 1e-6

    parameters_summary = tabulate(
        [
            [get_human_readable_count(trainable_params), "Trainable Params"],
            [get_human_readable_count(non_trainable_params), "Non-trainable Params"],
            [f"{all_params_megabytes:,.3f}", "Total estimated model params size (MB)"],
            [f"{all_buffers_megabytes:,.3f}", "Total estimated model buffers size (MB)"],
        ],
        tablefmt="plain",
        colalign=["right", "left"]
    )

    print(model_summary + "\n" + parameters_summary + "\n")


def save_checkpoint(
    config: DictConfig,
    global_step: int,
    epoch: int,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> None:
    """Pass in epoch=-1 to save the last checkpoint"""
    ckpt_path = os.path.join(config.train.log_dir, "ckpts", f"ckpt.{'last' if epoch == -1 else epoch}.pt")
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "step": global_step,
            "epoch": config.train.total_epochs if epoch == -1 else epoch,
        },
        ckpt_path,
    )
    return ckpt_path
