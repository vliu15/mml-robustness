"""Assorted import utilities for multi-task learning"""

import json
import os
from typing import List

import numpy as np

# Perhaps place this somewhere else? Right now it's used in get_mtl_task_weights
SPURIOUS_EVAL_DIR = "./outputs/erm_spurious_eval"


def get_mtl_task_weights(mtl_weighting: str, task_pairing: List[str], alpha: float = 0.5):

    if mtl_weighting == "static_equal":
        task_weights = [1] * len(task_pairing)
        use_loss_balanced = "false"
        lbtw_alpha = 0

        return task_weights, use_loss_balanced, lbtw_alpha

    elif mtl_weighting == "static_delta":
        use_loss_balanced = "false"
        lbtw_alpha = 0

        ## get delta value for each task:attribute pair
        deltas = []
        for grouping in task_pairing:
            task = grouping.split(":")[0]
            attribute = grouping.split(":")[1]
            with open(os.path.join(SPURIOUS_EVAL_DIR, task, f"{task}_spurious_eval.json"), "r") as f:
                data = json.load(f)
                deltas.append(data[attribute])

        ## compute delta heuristic, scale by 100 to get back into a range that won't blow up in softmax
        deltas_np = np.array(deltas) / 100
        task_weights_np = np.exp(deltas_np) / np.sum(np.exp(deltas_np))
        task_weights = list(task_weights_np)

        return task_weights, use_loss_balanced, lbtw_alpha
    elif mtl_weighting == "dynamic":
        task_weights = [1] * len(task_pairing)
        use_loss_balanced = "true"
        lbtw_alpha = alpha

        return task_weights, use_loss_balanced, lbtw_alpha
