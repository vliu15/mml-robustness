import logging
import os

import hydra
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter

from train_common import train
from utils.init_modules import init_ddp_dataloaders, init_ddp_model, init_ema, init_logdir, init_optimizer, init_scheduler
from utils.train_utils import seed_all_rng

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def train_rank(rank, world_size, config):
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
        global_step = ckpt["step"] + 1
        epoch = ckpt["epoch"] + 1
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


@hydra.main(config_path="configs/", config_name="default")
def main(config):
    config = config.exp

    if config.train.n_gpus == -1:
        config.train.n_gpus = torch.cuda.device_count()
    assert config.train.n_gpus > 1, f"Must be using >1 gpus for DDP training. {config.train.n_gpus} specified."
    init_logdir(config)

    multiprocessing.spawn(train_rank, args=[config.train.n_gpus, config], nprocs=config.train.n_gpus, join=True)


if __name__ == "__main__":
    main()
