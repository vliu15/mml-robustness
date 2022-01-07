"""Contains the entire train function for both train.py and train_ddp.py."""

import json
import logging
import os
from collections import defaultdict
from copy import deepcopy
from typing import Iterable

import torch
import torch.distributed as distributed
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.init_modules import init_dataloaders, init_ema, init_model, init_optimizer, init_scheduler
from utils.train_utils import (
    accumulate_stats,
    barrier,
    get_top_level_summary,
    log_stats,
    save_checkpoint,
    seed_all_rng,
    to_device,
)

# On rice.stanford.edu, only older versions of pytorch are supported
try:
    from torch.cuda.amp import GradScaler
except ModuleNotFoundError:
    GradScaler = None

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def train_step(
    *,
    global_step: int,
    batch: Iterable[torch.Tensor],
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler = None,
    device: str,
    rank: int = 0,
    first_batch_loss: torch.Tensor = None,
):
    """Performs one step of forward pass, backpropagation, optimizer and EMA step"""
    batch = to_device(batch, device)
    optimizer.zero_grad()

    # Mixed precision: O1 forward pass, scale/clip gradients, schedule LR when no gradient overflow
    if config.train.fp16:
        with torch.cuda.amp.autocast(enabled=config.train.fp16):
            scaling_factor = scaler.get_scale()

            loss_dict, metrics_dict = model.supervised_step(
                batch, subgroup=config.dataset.subgroup_labels, first_batch_loss=first_batch_loss
            )
            loss = loss_dict["loss"]
            if first_batch_loss is None and config.dataset.loss_based_task_weighting:
                first_batch_loss.data = loss_dict["first_batch_loss"].data
            scaler.scale(loss).backward()
            # Optionally apply gradient clipping
            if config.train.grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            # Check for gradient overflow, if none then schedule LR
            if scaling_factor == scaler.get_scale():
                scheduler.step()
            else:
                logger.debug("[Rank %s] Gradient overflow detected. Loss scale lowered to %s", rank, scaler.get_scale())
                scaling_factor = scaler.get_scale()

    # Full precision: O0 forward pass with optional gradient clipping, schedule LR
    else:
        loss_dict, metrics_dict = model.supervised_step(
            batch, subgroup=config.dataset.subgroup_labels, first_batch_loss=first_batch_loss
        )
        loss = loss_dict["loss"]
        if first_batch_loss is None and config.dataset.loss_based_task_weighting:
            first_batch_loss.data = loss_dict["first_batch_loss"].data
        if torch.isnan(loss):
            print(
                dict(
                    **{k: loss_dict[k] for k in loss_dict.keys() if k.startswith("loss")},
                    **metrics_dict,
                    STEP=global_step,
                    RANK=rank,
                )
            )
            raise RuntimeError(f"Nan detected in loss at step {global_step}")
        loss.backward()
        # Optionally apply gradient clipping
        if config.train.grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
        optimizer.step()
        # Schedule LR
        scheduler.step()

    ema.step()
    return loss_dict, metrics_dict


def train_epoch(
    *,
    global_step: int,
    epoch: int,
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    scaler: GradScaler = None,
    device: str,
    rank: int = 0,
):
    """Runs one epoch of standard neural network training"""
    postfix = {}
    losses, metrics = defaultdict(float), defaultdict(float)
    first_batch_loss = None

    # Train epoch
    with tqdm(
            total=len(train_dataloader),
            leave=False,
            desc=f"Epoch {epoch} [train]",
            disable=(rank != 0),
    ) as pbar:

        model.train()
        for batch in train_dataloader:
            batch = to_device(batch, device)

            loss_dict, metrics_dict = train_step(
                global_step=global_step,
                batch=batch,
                config=config,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                rank=rank,
                first_batch_loss=first_batch_loss,
            )

            # Update per-rank stepwise averages
            global_step = global_step + 1
            pbar.update(1)

            # [Rank 0] Update loss averages, progress bars
            if rank == 0:
                # Update averages
                accumulate_stats(
                    loss_dict=loss_dict,
                    metrics_dict=metrics_dict,
                    accumulated_loss=losses,
                    accumulated_metrics=metrics,
                    over_n_examples=config.train.log_every_n_steps,
                    batch_size=1,  # we can just average metrics per batch in training
                )

                # Log and update progress bars
                if global_step % config.train.log_every_n_steps == 0:
                    log_stats(
                        step_or_epoch=global_step,
                        writer=writer,
                        split="train",
                        losses=losses,
                        metrics=metrics,
                    )
                    postfix = dict(**losses, **metrics, lr=optimizer.param_groups[0]["lr"])
                    pbar.set_postfix(postfix)
                    losses, metrics = defaultdict(float), defaultdict(float)

    return global_step, epoch


def val_step(
    *,
    batch: Iterable[torch.Tensor],
    config: DictConfig,
    model: nn.Module,
    device: str,
):
    """Performs one validation step"""
    batch = to_device(batch, device)
    loss_dict, metrics_dict = model.supervised_step(batch, subgroup=config.dataset.subgroup_labels, first_batch_loss=None)
    return loss_dict, metrics_dict


def val_epoch(
    *,
    epoch: int,
    config: DictConfig,
    model: nn.Module,
    ema: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    writer: torch.utils.tensorboard.SummaryWriter,
    device: str,
):
    """Runs one epoch of validation"""
    losses, metrics = defaultdict(float), defaultdict(float)

    with torch.no_grad():
        model.eval()
        ema.swap()

        for batch in tqdm(
                val_dataloader,
                total=len(val_dataloader),
                leave=False,
                desc=f"Epoch {epoch} [val]",
        ):
            loss_dict, metrics_dict = val_step(
                batch=batch,
                config=config,
                model=model,
                device=device,
            )

            # Accumulate losses, ground truths, and predictions
            accumulate_stats(
                loss_dict=loss_dict,
                metrics_dict=metrics_dict,
                accumulated_loss=losses,
                accumulated_metrics=metrics,
                over_n_examples=len(val_dataloader.dataset),
                batch_size=batch[0].shape[0],
            )
        ema.swap()

    log_stats(
        step_or_epoch=epoch,
        writer=writer,
        split="val",
        losses=losses,
        metrics=metrics,
    )
    postfix = {**losses, **metrics}
    return postfix


def train(
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
):
    # Check that model has proper handle
    assert hasattr(model, "supervised_step") and callable(model.supervised_step), \
        f"Model type {model.__class__.__name__} doesn't have forward handle `supervised_step`"

    barrier()
    if rank == 0:
        get_top_level_summary(model)

    # Additional global variables
    scaler = GradScaler() if config.train.fp16 else None

    # Train
    postfix, all_val_stats = {}, {}
    with tqdm(initial=epoch, total=config.train.total_epochs, desc="Global epoch", postfix=postfix,
              disable=(rank != 0)) as pbar:

        if config.train.run_sanity_val_epoch and rank == 0:
            logger.info("Running sanity val epoch")
            postfix = val_epoch(
                epoch=epoch,
                config=config,
                model=model,
                ema=ema,
                val_dataloader=val_dataloader,
                writer=writer,
                device=device,
            )
            pbar.set_postfix(postfix)
            with open(os.path.join(config.train.log_dir, "val_sanity.json"), "w") as f:
                json.dump(postfix, f)
            logger.info("Sanity val epoch done: %s", postfix)

        # Loop through epochs
        while True:
            if epoch >= config.train.total_epochs:
                break

            global_step, epoch = train_epoch(
                global_step=global_step,
                epoch=epoch,
                config=config,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=train_dataloader,
                writer=writer,
                scaler=scaler,
                device=device,
                rank=rank,
            )
            if epoch % config.train.eval_every_n_epochs == 0 and rank == 0:
                postfix = val_epoch(
                    epoch=epoch,
                    config=config,
                    model=model,
                    ema=ema,
                    val_dataloader=val_dataloader,
                    writer=writer,
                    device=device,
                )
                all_val_stats[epoch] = deepcopy(postfix)

            barrier()
            pbar.set_postfix(postfix)
            pbar.update(1)
            epoch = epoch + 1

            # Save checkpoint
            if global_step % config.train.ckpt_every_n_epochs == 0:
                save_checkpoint(config, global_step, epoch, model, ema, optimizer, scheduler)

    if rank == 0:
        save_checkpoint(config, global_step, -1, model, ema, optimizer, scheduler)
        writer.close()

        with open(os.path.join(config.train.log_dir, "val_stats.json"), "w") as f:
            json.dump(all_val_stats, f)


## End of training helpers, train_multi for ddp and train_single for single xpu


def train_ddp(rank, world_size, config):
    """Entry point into ddp training"""
    # RNG
    seed_all_rng(config.seed, cuda=True)

    # Initialize rank process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    distributed.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    device = torch.device(rank)
    torch.cuda.set_device(rank)

    # Init modules
    writer = SummaryWriter(config.train.log_dir) if rank == 0 else None
    model = init_model(config, device=device)
    ema = init_ema(config, model)
    optimizer = init_optimizer(config, model)
    scheduler = init_scheduler(config, optimizer)
    train_dataloader, val_dataloader = init_dataloaders(config)

    # Wait here in case rank=0 is running DDI, then load config
    distributed.barrier()
    config = OmegaConf.load(os.path.join(config.train.log_dir, "config.yaml"))

    # Load checkpoint
    if config.train.load_ckpt:
        ckpt = torch.load(config.train.load_ckpt, map_location=torch.device(rank))
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        ema.load_state_dict(ckpt["ema"])
        global_step = ckpt["step"]
        epoch = ckpt["epoch"]
    else:
        global_step = 0
        epoch = 0

    logger.info(" [Rank %s / %s] Initialized all modules", rank, world_size)

    # Train
    try:
        distributed.barrier()
        train(
            global_step=global_step,
            epoch=epoch,
            config=config,
            model=model.module,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            writer=writer,
            device=device,
            rank=rank,
        )
    except KeyboardInterrupt:
        pass

    # Destroy rank process
    distributed.destroy_process_group()


def train_single(config):
    """Entry point into single xpu training"""
    # RNG
    cuda = torch.cuda.is_available()
    seed_all_rng(config.seed, cuda=cuda)

    device = torch.device("cuda") if cuda else torch.device("cpu")

    # Init modules
    writer = SummaryWriter(config.train.log_dir)
    model = init_model(config, device=device)
    ema = init_ema(config, model)
    optimizer = init_optimizer(config, model)
    scheduler = init_scheduler(config, optimizer)
    train_dataloader, val_dataloader = init_dataloaders(config)

    # Load checkpoint
    if config.train.load_ckpt:
        ckpt = torch.load(config.train.load_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        ema.load_state_dict(ckpt["ema"])
        global_step = ckpt["step"]
        epoch = ckpt["epoch"]
    else:
        global_step = 0
        epoch = 0

    logger.info("[%s] Initialized all modules", device)

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
        rank=0,
    )
