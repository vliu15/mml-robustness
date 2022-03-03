"""
Script to create and output task pairings based on identified spurious correlations.
Generates all possible pairigns that exist for a given specification and will take a random sample 
to provide the user specified amount back, all pairings, or none if no task pairings exist

python -m scripts.task_pairing_identification \
    --spurious_eval_dir ./outputs/suby_spurious_eval  \
    --spurious_epsilon 30 \
    --out_dir ./outputs/suby_task_pairings\
    --pairing_type disjoint \
    --svd_pairings None \
    --num_tasks 2 \
    --num_pairings 5 \
"""

import argparse
import datasets.groupings as groupings
from create_spurious_matrix import attributes

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
        help="The epsilon value with which to define spurious correaltes per task by"
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
        "--svd_pairings",
        type=str,
        default=None,
        required=False,
        help="A json file of common clusterings as given by SVD/Biclustering"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=2,
        required=True,
        help="The number of tasks to include in each pairing"
    )
    parser.add_argument(
        "--num_pairings",
        type=int,
        default=5,
        required=True,
        help="The number of pairings to return"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    ### for all attributes, store all spurious correlations dict of attribute to set of spurious correlations
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load JSON files for each task
    T = []
    for task in tqdm(attributes, desc="Loading JSONs", total=len(attributes)):
        with open(os.path.join(args.spurious_eval_dir, task, f"{task}_spurious_eval.json"), "r") as f:
            data = json.load(f)
            column = np.array([data[attr] for attr in attributes], dtype=np.float32)
            T.append(column)

    # Also define average performance above

    # Vectorize and Define Spurious Correlations

    T = np.array(T)  # shape (40, 40) that corresponds to (task, attr)
    T = T >= args.spurious_epsilon

    ## i

    task_to_correlates = {}

    ### find all groupings of task

if __name__ == "__main__":
    main()
