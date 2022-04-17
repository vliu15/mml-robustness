"""
Script to create and output task pairings based on identified spurious correlations.
Generates all possible pairigns that exist for a given specification and will take a random sample 
to provide the user specified amount back, all pairings, or none if no task pairings exist

python -m scripts.task_pairing_identification \
    --spurious_eval_dir ./outputs/suby_spurious_eval  \
    --spurious_epsilon 30 \
    --out_dir ./outputs/suby_task_pairings\
    --pairing_type disjoint \
    --svd_clusters None \
    --num_tasks 2 \
    --num_pairings 5 \
"""

import argparse
import itertools
import json
import os
import random

import numpy as np
from tqdm import tqdm

import datasets.groupings as groupings
from scripts.spurious_matrix import attributes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spurious_eval_dir",
        type=str,
        default="./",
        required=True,
        help="The folder which contains all results from spurious id"
    )
    parser.add_argument(
        "--spurious_epsilon",
        type=int,
        default=30,
        required=True,
        help="The epsilon value with which to define spurious correlates per task by"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/task_pairings",
        required=True,
        help="Where to output the results task pairings to"
    )
    parser.add_argument(
        "--pairing_type",
        type=str,
        default="disjoint",
        required=False,
        choices=["disjoint", "nondisjoint", "ndpoor", "dpoor"],
        help="Whether to port the tuning results or the heatplot results"
    )
    parser.add_argument(
        "--svd_clusters",
        type=str,
        default=None,
        required=False,
        help="A json file of common clusterings as given by SVD/Biclustering"
    )
    parser.add_argument(
        "--num_tasks", type=int, default=2, required=True, help="The number of tasks to include in each pairing"
    )
    parser.add_argument("--num_pairings", type=int, default=5, required=True, help="The number of pairings to return")

    args = parser.parse_args()

    return args


def main(attributes):
    args = parse_args()

    ### for all attributes, store all spurious correlations dict of attribute to set of spurious correlations
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.num_tasks > len(attributes):
        raise ValueError(
            f"There are only {len(attributes)} possible tasks but the user provided a num tasks value greater than this."
        )

    # Load JSON files for each task
    T = []
    task_performance = []
    for task in tqdm(attributes, desc="Loading JSONs", total=len(attributes)):
        with open(os.path.join(args.spurious_eval_dir, task, f"{task}_spurious_eval.json"), "r") as f:
            data = json.load(f)
            column = np.array([data[attr] for attr in attributes], dtype=np.float32)
            T.append(column)

        with open(os.path.join(args.spurious_eval_dir, task, f"{task}:{task}.json"), "r") as f:
            data = json.load(f)
            task_performance.append(data[f"{task}_avg_acc"])

    svd_clusters = None
    if args.svd_clusters is not None:
        with open(args.svd_clusters, "r") as f:
            svd_clusters = json.load(f)  ## dict

    # Vectorize and Define Spurious Correlations

    T = np.array(T)  # shape (40, 40) that corresponds to (task, attr)
    T = T >= args.spurious_epsilon

    task_performance = np.array(task_performance)
    task_performance = task_performance < np.percentile(task_performance, 33)

    ## deliniate correlates for each task and poor performing tasks
    poor_tasks = []
    task_to_correlates = {}

    poor_tasks = [attributes[ind] for ind in np.where(task_performance == True)[0]]
    for ind, attr in enumerate(attributes):
        row = T[ind]
        spurious_correlates = set([attributes[ind] for ind in np.where(row == True)[0]])

        if args.svd_clusters is not None:
            correlates_to_add = set()
            for _, cluster_members in svd_clusters.items():
                for correlate in spurious_correlates:
                    if correlate in cluster_members:
                        for new_correlate in cluster_members:
                            correlates_to_add.add(new_correlate)

            spurious_correlates.update(correlates_to_add)

        task_to_correlates[attr] = spurious_correlates

    # find all groupings of tasks in the desired quantity in a pairing

    if args.pairing_type in ["ndpoor", "dpoor"]:
        attributes = poor_tasks
        if args.num_tasks > len(poor_tasks):
            raise ValueError(
                f"There are only {len(poor_tasks)} possible poor performing tasks but the user provided a num tasks value greater than this."
            )

    possible_task_pairings = list(itertools.combinations(attributes, args.num_tasks))

    if len(possible_task_pairings) > 1e6:
        possible_task_pairings = random.sample(possible_task_pairings, int(1e6))

    ## find all possible pairings that satisfy constraints
    possible_tasks = []  ### should be a dict of key to list that we can form groupings from

    for pairing in possible_task_pairings:

        ### we want that pairwise intersection is empty
        if args.pairing_type in ["disjoint", 'dpoor']:
            correlate_set_computation = task_to_correlates[pairing[0]]
            num_sets = len(correlate_set_computation)
            if num_sets != 0:
                for task in pairing[1:]:
                    correlates = task_to_correlates[task]
                    if len(correlates) == 0:
                        num_sets = -1
                        break
                    num_sets += len(correlates)
                    correlate_set_computation = correlate_set_computation.union(correlates)

                if num_sets == len(correlate_set_computation):
                    possible_task = {}
                    for task in pairing:
                        possible_task[task] = random.sample([*task_to_correlates[task]], 1)

                    possible_tasks.append(possible_task)

        ### we want that their intersection is not empty and we will use it
        elif args.pairing_type in ["nondisjoint", 'ndpoor']:
            correlate_set_computation = task_to_correlates[pairing[0]]
            for task in pairing[1:]:
                correlates = task_to_correlates[task]
                correlate_set_computation = correlate_set_computation.intersection(correlates)

            if len(correlate_set_computation) > 0:
                possible_task = {}
                for task in pairing:
                    possible_task[task] = random.sample([*correlate_set_computation], 1)

                possible_tasks.append(possible_task)

    ### sample the number of pairings we want to return if 0 return none, if less than return all

    if len(possible_tasks) == 0:
        raise ValueError("There are no task pairings which satisfy the specified constraints")

    chosen_task_pairings = possible_tasks
    if len(possible_tasks) > args.num_pairings:
        chosen_task_pairings = random.sample(possible_tasks, args.num_pairings)

    ## greate MTL groupings from each, serialize into list of json strings to save
    json_serialize = []
    for task_pairing in chosen_task_pairings:
        task_groupings = []
        for task, spurious in task_pairing.items():
            task_groupings.append(groupings.Grouping(task_label=task, subgroup_attributes=spurious))

        mtl_grouping = groupings.MTLGrouping(*task_groupings)
        mtl_json_string = repr(mtl_grouping)
        json_serialize.append(mtl_json_string)

    task_pairing_file = os.path.join(args.out_dir, f"{args.pairing_type}_task_pairings_n_tasks_{args.num_tasks}.json")
    f = open(task_pairing_file, "w")
    json.dump(json_serialize, f)
    f.close()


if __name__ == "__main__":
    main(attributes)
