"""Implements instance weighting via convex optimization"""

import logging
import logging.config
import time

import cvxpy as cp
import numpy as np
import scipy


def quadratic_programming(Y, verbose=False):
    """
    Solves the constrained least-squares / quadratic program objective:

        \min_w 0.5 * ||Yw - c||_2^2 \\
        s.t. \begin{cases}
            \sum_i w_i = 1 \\
            w_i \ge 0
        \end{cases}

    where c is a w-sized constant vector of values 1/2.

    Reference: https://www.cvxpy.org/examples/basic/quadratic_program.html
    """
    k, n = Y.shape  # k = number of tasks, n = number of instances
    c = 0.5 * np.ones((k,), dtype=np.float32)
    print("Solving for w over %d examples. Input array of labels: (%d tasks, %d examples).", n, k, n)

    # Solve
    w = cp.Variable(n, nonneg=True)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(w, Y.T @ Y) - np.dot(c @ Y, w)),  # quadratic program objective
        [
            cp.sum(w) == 1,  # w as a distribution constraint
        ]
    )

    start = time.time()
    prob.solve(solver=cp.ECOS, verbose=verbose)
    end = time.time()
    print("Took %.2f seconds to solve the quadratic program.", (end - start))

    return w.value


def entropy_maximization(Y, verbose=False):
    """
    Solves the linearly inequality constrained entropy maximization problem:

        \min_w -\sum_i w_i \log(w_i) \\
        s.t. \begin{cases}
            Yw = c \\
            \sum_i w_i = 1 \\
            s_i \ge 0
        \end{cases}

    where c is a w-sized constant vector of values 1/2.

    Reference: https://www.cvxpy.org/examples/applications/max_entropy.html
    """
    Y = Y.astype(np.float32)
    k, n = Y.shape  # k = number of tasks, n = number of instances
    c = 0.5 * np.ones((k,), dtype=np.float32)
    print(f"Solving for w over {n} examples. Input array of labels: ({k} tasks, {n} examples).")

    # Solve
    w = cp.Variable(n, nonneg=True)
    prob = cp.Problem(
        cp.Maximize(cp.sum(cp.entr(w))),
        [
            Y @ w == c,
            cp.sum(w) == 1,
        ]
    )

    start = time.time()
    prob.solve(solver=cp.ECOS, verbose=verbose, qcp=True)
    print("Caught ECOS error, probably due to unconverged tolerance")
    end = time.time()

    print(f"Took {end - start:.2f} seconds to solve the entropy minimization problem.")
    print(w.value)
    return w.value


if __name__ == "__main__":
    # This is just a code block to test out the optimization functions independent of the train scripts.
    from omegaconf import OmegaConf
    config = OmegaConf.load("./configs/exp/erm.yaml")
    config.dataset.groupings = [
        "Big_Lips:Chubby",
        "Bushy_Eyebrows:Blond_Hair",
        "Wearing_Lipstick:Male",
        "Gray_Hair:Young",
        "High_Cheekbones:Smiling",
        "Goatee:No_Beard",
        "Wavy_Hair:Straight_Hair",
    ]
    config.dataset.task_weights = [1, 1, 1, 1, 1, 1, 1]

    # Initialize train dataset
    from datasets.celeba import CelebA
    dataset = CelebA(config, split="train")
    Y = dataset.attr[:, dataset.task_label_indices].T.numpy()  # (7, 162770)
    Y = scipy.sparse.csr_matrix(Y)
    # w = quadratic_programming(Y, verbose=True)
    w = entropy_maximization(Y, verbose=True)
