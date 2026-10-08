"""Optional interface for Random Search in the common forest space.

Implement this loop, connect a package, or use another organisation. Choose how
to retain the results needed to analyse search progress and computational effort.
"""

from __future__ import annotations
from typing import Any
from random_forest import Config, Evaluator, sample_configuration
import numpy as np


def optimise_random_search(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: randomly sample and evaluate up to n_trials configurations.

    Use the shared search space and train each forest with n_trees trees.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """
    best_config = None
    best_score = -1 # accuracy always higher than -1
    results = [] # here for each trail we save the eval_dict in case we want to return it as well or look at info

    rng = np.random.default_rng(seed=seed)

    # n_trails attempt are made to randomly design it
    for trail in range(n_trials):
        attempt_config = sample_configuration(rng)
        # use the evaluator to get the metrics
        eval_dict = evaluator(attempt_config, n_trees, seed)
        results.append(eval_dict)

        # compare objective with saved score, highest score is the best config
        if best_score < eval_dict["objective"]:
            best_score = eval_dict["objective"]
            best_config = eval_dict["configuration"]

    # print(f"best found config: {best_config} ")
    # print(f"best found score: {best_score}")

    # return the best found config with its score
    return best_config, best_score, results  

