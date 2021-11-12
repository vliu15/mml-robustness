import json
from collections import defaultdict

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from train_common import to_device
from utils.init_modules import init_logdir, init_model, init_test_dataloader
from utils.train_utils import seed_all_rng


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
    model = init_model(config).to(device)
    test_dataloader = init_test_dataloader(config)

    # Load checkpoint
    ckpt = torch.load(config.test.load_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    evaluate(
        config=config,
        model=model,
        test_dataloader=test_dataloader,
        device=device,
    )


def evaluate(config: DictConfig, model: nn.Module, test_dataloader: torch.utils.data.DataLoader, device: str):

    test_stats = defaultdict(float)
    for batch in tqdm(test_dataloader, total=len(test_dataloader.dataset)):
        with torch.no_grad():

            batch = to_device(batch, device)
            out_dict = model.supervised_step(batch, subgroup=config.dataset.subgroup_labels)

            for key in out_dict.keys():

                if key.startswith("metric"):

                    if "avg" in key:
                        test_stats[key] += out_dict[key].item() * test_dataloader.batch_size
                    elif "count" in key:
                        pass
                    else:

                        past_acc = test_stats[key]
                        count_key = key + "_count"
                        past_count = test_stats[count_key]

                        if (past_count + out_dict[count_key]) != 0:
                            updated_acc = (past_acc * past_count + out_dict[key].item() * out_dict[count_key]) / (
                                past_count + out_dict[count_key]
                            )
                        else:
                            updated_acc = 0.0
                        test_stats[count_key] += out_dict[count_key]
                        test_stats[key] = updated_acc

    for key in test_stats.keys():
        if key.startswith("metric"):
            if "avg" in key:
                test_stats[key] = test_stats[key] / len(test_dataloader.dataset)

    print("Test Set Results:")
    print(test_stats)

    log_dir = str(config.test.results_dir) + "test_results.json"
    with open(log_dir, 'w') as fp:
        json.dump(test_stats, fp)


if __name__ == "__main__":
    main()
