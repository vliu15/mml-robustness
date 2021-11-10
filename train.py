import logging

import hydra
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.init_modules import init_dataloaders, init_ema, init_logdir, init_model, init_optimizer, init_scheduler
from utils.train_utils import seed_all_rng

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="default")
def main(config):
    """Entry point into single XPU training."""
    config = config.exp

    # RNG
    cuda = torch.cuda.is_available()
    seed_all_rng(config.seed, cuda=cuda)

    init_logdir(config)
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

    train_fn = None
    if config.train.name == "jtt":
        from train_jtt import train_jtt
        train_fn = train_jtt
    elif config.train.name == "erm":
        from train_common import train
        train_fn = train
    else:
        raise ValueError(f"Unrecognized train.name: {config.train.name}")

    train_fn(
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


if __name__ == "__main__":
    main()
