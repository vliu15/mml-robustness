"""Script for standard ERM training"""

import logging
import os

import hydra
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter

from train_common import train
from utils.init_modules import (
    init_dataloaders,
    init_ddp_dataloaders,
    init_ddp_model,
    init_ema,
    init_logdir,
    init_model,
    init_optimizer,
    init_scheduler,
)
from utils.train_utils import seed_all_rng

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def train_ddp(rank, world_size, config):
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
    model = init_ddp_model(config, rank)
    ema = init_ema(config, model)
    optimizer = init_optimizer(config, model)
    scheduler = init_scheduler(config, optimizer)
    train_dataloader, val_dataloader = init_ddp_dataloaders(config, rank, world_size)

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

    logger.info(f" [Rank {rank} / {world_size}] Initialized all modules")

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
    # RNG
    cuda = torch.cuda.is_available()
    seed_all_rng(config.seed, cuda=cuda)

    device = torch.device("cuda") if cuda else torch.device("cpu")
    config.train.n_gpus = 1 if cuda else 0

    # Init modules
    writer = SummaryWriter(config.train.log_dir)
    model = init_model(config).to(device)
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

    logger.info(f"[{device}] Initialized all modules")

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


@hydra.main(config_path="configs/", config_name="default")
def main(config):
    """Entry point into ERM training"""
    config = config.exp
    init_logdir(config)

    max_gpus = torch.cuda.device_count()
    if config.train.n_gpus == -1:
        config.train.n_gpus = max_gpus


    ## make sure task weights are correctly specified
    if len(config.dataset.task_weights) != len(config.dataset.task_labels):
        raise ValueError("Task weights must be the same length as task labels")

    if sum(config.dataset.task_weights) != 1:
        if len(set(config.dataset.task_weights)) != 1 or 1 not in config.dataset.task_weights:
            raise ValueError("Ensure task weights are either all 1 (no weighting) or sum up to 1 (weighting)")

    if config.dataset.loss_based_task_weighting:
        if len(set(config.dataset.task_weights)) != 1 or 1 not in config.dataset.task_weights:
            raise ValueError("To apply loss based task weighting, the original task weights must be all 1")
        
        
    # Determine whether to launch single or multi-gpu training
    n_gpus = min(config.train.n_gpus, max_gpus)
    if n_gpus <= 1:
        train_single(config)
    else:
        assert config.train.n_gpus <= max_gpus, f"Specified {config.train.n_gpus} gpus, but only {max_gpus} total"
        multiprocessing.spawn(train_ddp, args=[config.train.n_gpus, config], nprocs=config.train.n_gpus, join=True)


if __name__ == "__main__":
    main()
