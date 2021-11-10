"""File containing init handles for various modules"""
# All lazy imports to avoid importing unnecessary modules


def init_model(config):
    if config.model.name == "resnet50":
        from models.resnet import ResNet50
        return ResNet50(config)
    else:
        raise ValueError(f"Didn't recognize model name {config.model.name}")


def init_ddp_model(config, rank):
    model = init_model(config).to(rank)

    # Check for possible conversion to synchronized batchnorm
    from torch.nn.modules.batchnorm import _BatchNorm
    if any(isinstance(module, _BatchNorm) for module in model.modules()):
        from torch.nn import SyncBatchNorm
        model = SyncBatchNorm.convert_sync_batchnorm(model)

    from torch.nn.parallel import DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=True)
    return model


def init_datasets(config):
    if config.dataset.name == "celeba":
        from datasets.celeba import CelebA
        dataset = CelebA
    else:
        raise ValueError(f"Didn't recognize dataset name {config.dataset.name}")

    return dataset(config, split='train'), dataset(config, split='val')


def init_dataloaders(config):
    train_dataset, val_dataset = init_datasets(config)

    from torch.utils.data import DataLoader
    return (
        DataLoader(
            train_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=True,
            pin_memory=True,
        ),
        DataLoader(
            val_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            shuffle=False,
            pin_memory=True,
        ),
    )


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
        shuffle=True,
        pin_memory=True,
    )


def init_ddp_dataloaders(config, rank, world_size):
    train_dataset, val_dataset = init_datasets(config)

    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True),
        pin_memory=True,
    )
    if rank == 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
        )
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader


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
