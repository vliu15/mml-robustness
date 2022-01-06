"""File containing init handles for various modules"""
# All lazy imports to avoid importing unnecessary modules

import logging

import torch.distributed as distributed

logging.config.fileConfig("logger.conf")
logger = logging.getLogger(__name__)


def init_model(config, device="cuda"):
    if config.model.name == "resnet50":
        from models.resnet import ResNet50
        model = ResNet50(config)
    else:
        raise ValueError(f"Didn't recognize model name {config.model.name}")

    model = model.to(device)

    # Wrap model in DDP if initialized
    if distributed.is_initialized():
        # Convert BN -> SyncBN if applicable
        from torch.nn.modules.batchnorm import _BatchNorm
        if any(isinstance(module, _BatchNorm) for module in model.modules()):
            from torch.nn import SyncBatchNorm
            model = SyncBatchNorm.convert_sync_batchnorm(model)

        from torch.nn.parallel import DistributedDataParallel
        rank = distributed.get_rank()
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=True)

    return model


def init_datasets(config):
    if config.dataset.name == "celeba":
        from datasets.celeba import CelebA
        dataset = CelebA
    else:
        raise ValueError(f"Didn't recognize dataset name {config.dataset.name}")

    # Upweight/upweight training dataset if provided
    if config.train.get("up_type", None):
        if config.train.up_type == "upsample":
            from datasets.upsample import UpsampledDataset
            return (
                UpsampledDataset(dataset(config, split='train'), config.train.load_up_pkl),
                dataset(config, split='val'),
            )
        elif config.train.up_type == "upweight":
            raise ValueError(f"up_type `upweight` is deprecated")
            # from datasets.upweight import UpweightedDataset
            # return (
            #     UpweightedDataset(dataset(config, split='train'), config.train.lambda_up, config.train.load_up_pkl),
            #     dataset(config, split='val'),
            # )
        else:
            raise ValueError(f"Didn't recognize up_type {config.train.up_type}")

    return dataset(config, split='train'), dataset(config, split='val')


def init_dataloaders(config):
    """Returns train and val dataloaders (val only for rank==0)"""
    train_dataset, val_dataset = init_datasets(config)

    # Init (DDP) train dataloader
    from torch.utils.data import DataLoader
    if distributed.is_initialized():
        world_size = distributed.get_world_size()
        rank = distributed.get_rank()
        from torch.utils.data.distributed import DistributedSampler
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True),
            pin_memory=True,
            drop_last=False,
        )
    else:
        rank = 0
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    # Init (Rank 0) val dataloader
    if rank == 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader


def init_test_dataset(config):
    if config.dataset.name == "celeba":
        from datasets.celeba import CelebA
        dataset = CelebA
    else:
        raise ValueError(f"Didn't recognize dataset name {config.dataset.name}")

    return dataset(config, split='test')


def init_test_dataloader(config):
    test_dataset = init_test_dataset(config)

    from torch.utils.data import DataLoader
    return DataLoader(
        test_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )


def init_optimizer(config, model):
    if config.optimizer.name == "adam":
        from torch.optim import AdamW
        return AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=config.optimizer.betas,
            weight_decay=config.optimizer.weight_decay,
            eps=config.optimizer.eps,
        )
    elif config.optimizer.name == "sgd":
        from torch.optim import SGD
        return SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Didn't recognize optimizer name {config.optimizer.name}")


def init_scheduler(config, optimizer):
    if not config.get("scheduler", None):
        from utils.lr_scheduler import DummyLR
        return DummyLR(optimizer)
    elif config.scheduler.name == "noam":
        from utils.lr_scheduler import NoamLR
        return NoamLR(
            optimizer,
            dim_model=config.model.encoder.hidden_channels,
            warmup_steps=config.scheduler.warmup_steps,
        )
    elif config.scheduler.name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=config.train.total_steps)
    elif config.scheduler.name == "linear":
        from utils.lr_scheduler import LinearWarmupLR
        return LinearWarmupLR(optimizer, warmup_steps=config.scheduler.warmup_steps)
    else:
        raise ValueError(f"Didn't recognize scheduler name {config.scheduler.name}")


def init_ema(config, model):
    if not config.train.get("ema", False):
        from models.ema import DummyEMA
        return DummyEMA()
    else:
        from models.ema import EMA
        return EMA(model, mu=1 - (config.dataloader.batch_size * config.train.n_gpus / 1000.))


def init_logdir(config):
    import os

    from hydra.utils import to_absolute_path
    from omegaconf import OmegaConf
    config.train.log_dir = to_absolute_path(config.train.log_dir)
    os.makedirs(config.train.log_dir, exist_ok=True)
    os.makedirs(os.path.join(config.train.log_dir, "ckpts"), exist_ok=True)
    with open(os.path.join(config.train.log_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=config, f=f.name)

    logger.info(f"Set up logdir at {config.train.log_dir}")
