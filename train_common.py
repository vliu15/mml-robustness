"""Contains the entire train function for both train.py and train_ddp.py."""

import logging
import os
from collections import defaultdict

import torch
import torch.distributed as distributed
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from utils.train_utils import get_top_level_summary, save_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def to_device(batch, device):
    """Puts a batch onto the specified device"""
    return [b.to(device) for b in batch if isinstance(b, torch.Tensor)]


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
    exp: str = "",
) -> None:
    """
    Trains the model. A few rules of thumb:
    - For single xpu training, simply pass in rank=0
    - Only rank 0 prints to console, logs, ckpts, and runs eval
    - Optional training features: model EMA, LR scheduler, mixed precision, gradient clipping

    Args:
        global_step: starting step in training
        epoch: starting epoch in training
        config: dict config that specifies all training parameters
        ema: module to optionally track exponential moving average of weights
        optimizer: optimizer to perform gradient descent on the loss
        scheduler: scheduler to optionally scale the optimizer learning rates
        writer: (tensorboard) logger to track spectrograms, audio samples, and losses
        device: string name of device to put this training process on
        rank: index that the device corresponds to with (respect to the world size)
        exp: any string prefix to organize logs
    """
    # Strip "/" so we can use this to prefix logs
    exp = exp.strip("/")

    # Handy function for synchronizing processes in multi-gpu training
    def barrier():
        if distributed.is_initialized():
            distributed.barrier()

    # Print table of model summary
    barrier()
    if rank == 0:
        get_top_level_summary(model)

    # Set up gradient scaler if training in mixed precision
    if config.train.fp16:
        scaler = torch.cuda.amp.GradScaler()
        scaling_factor = scaler.get_scale()

    # Train
    postfix = {}
    train_stats = defaultdict(float)
    with tqdm(initial=global_step, total=config.train.total_steps, desc="Global step", postfix=postfix) as pbar:

        while True:
            model.train()
            for batch in tqdm(
                    train_dataloader,
                    total=len(train_dataloader),
                    leave=False,
                    desc=f"Epoch {epoch} [train]",
                    disable=(rank != 0),
            ):
                batch = to_device(batch, device)
                optimizer.zero_grad()

                # Mixed precision: O1 forward pass, scale/clip gradients, schedule LR when no gradient overflow
                if config.train.fp16:
                    with torch.cuda.amp.autocast():
                        out_dict = model.supervised_step(batch)
                        loss = out_dict["loss"]
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
                            logger.debug(
                                f"[Rank {rank}] Gradient overflow detected. Loss scale lowered to {scaler.get_scale()}"
                            )
                            scaling_factor = scaler.get_scale()

                # Full precision: O0 forward pass with optional gradient clipping, schedule LR
                else:
                    out_dict = model.supervised_step(batch)
                    loss = out_dict["loss"]
                    if torch.isnan(loss):
                        logger.info(
                            dict(
                                **{k: out_dict[k] for k in out_dict.keys() if k.startswith("loss")},
                                STEP=global_step,
                                EPOCH=epoch,
                                RANK=rank,
                            )
                        )
                        raise RuntimeError(f"Nan detected in loss at step {global_step}, epoch {epoch}")
                    loss.backward()
                    # Optionally apply gradient clipping
                    if config.train.grad_clip_norm:
                        nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                    optimizer.step()
                    # Schedule LR
                    scheduler.step()

                # Update per-rank stepwise averages
                ema.step()
                global_step += 1

                # [Rank 0] Update loss averages, progress bars, and checkpoint
                if rank == 0:
                    # Update loss average
                    for key in out_dict.keys():
                        if key.startswith("loss") or key.startswith("metric"):
                            train_stats[key] += out_dict[key].item()

                    # Log losses/metrics and update progress bars
                    if global_step % config.train.log_every_n_steps == 0:
                        for key in train_stats.keys():
                            train_stats[key] /= config.train.log_every_n_steps
                            if key.startswith("loss"):
                                writer.add_scalar(os.path.join(exp, "loss", f"train_{key}"), train_stats[key], global_step)
                            elif key.startswith("metric"):
                                writer.add_scalar(os.path.join(exp, "metric", f"train_{key}"), train_stats[key], global_step)
                        postfix = dict(**train_stats, lr=optimizer.param_groups[0]["lr"])
                        pbar.set_postfix(postfix)
                        train_stats = defaultdict(float)
                        logger.debug(dict(**postfix, STEP=global_step, EPOCH=epoch))

                    # Checkpoint
                    if global_step % config.train.ckpt_every_n_steps == 0:
                        save_checkpoint(config, global_step, epoch, model, ema, optimizer, scheduler)
                    pbar.update(1)

                # Check if training is done
                if global_step >= config.train.total_steps:
                    return

            # [Rank 0] Run evaluation with EMA, save ground truth and predictions for comparison
            if rank == 0 and epoch % config.train.eval_every_n_epochs == 0:
                val_stats = defaultdict(float)

                with torch.no_grad():
                    model.eval()
                    ema.swap()
                    for batch in tqdm(
                            val_dataloader,
                            total=len(val_dataloader),
                            leave=False,
                            desc=f"Epoch {epoch} [val]",
                    ):
                        batch = to_device(batch, device)
                        with torch.cuda.amp.autocast(enabled=config.train.fp16):
                            out_dict = model.supervised_step(batch)

                        # Accumulate losses/metrics
                        for key in out_dict.keys():
                            if key.startswith("loss") or key.startswith("metric"):
                                val_stats[key] += out_dict[key].item() * val_dataloader.batch_size
                    ema.swap()

                # Log losses/metrics
                for key in val_stats.keys():
                    if key.startswith("loss"):
                        writer.add_scalar(
                            os.path.join(exp, "loss", f"val_{key}"), val_stats[key] / len(val_dataloader.dataset), epoch
                        )
                    elif key.startswith("metric"):
                        writer.add_scalar(
                            os.path.join(exp, "metric", f"val_{key}"), val_stats[key] / len(val_dataloader.dataset), epoch
                        )

                # Add additional post-evaluation logging here (i.e. images, audio, text)
                pass

            # End-of-epoch logistics
            epoch += 1
            barrier()

    # Save last checkpoint
    if rank == 0:
        save_checkpoint(config, global_step, epoch, model, ema, optimizer, scheduler)
