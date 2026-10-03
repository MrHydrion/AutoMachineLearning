"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any

from operator import itemgetter # used to sort based on score

from random_forest import Config, Evaluator, sample_configuration

import math

import numpy as np

def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.
    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """

    # Hyperband uses a number of brackets that depends on the available resource range
    # (here the minimum and maximum number of trees) and the reduction factor.
    # Based on Algorithm 1 from the original Hyperband paper:
    # https://www.jmlr.org/papers/volume18/16-558/16-558.pdf
    
    rng = np.random.default_rng(seed=seed)

    # used to save all dictionaries
    all_results = [] 

    
    R = max_trees    

    # / min_trees to make sure it does not fall below this value
    s_max = math.floor(math.log(x=R / min_trees, base=reduction_factor))

    B = (s_max + 1) * R
    #  we start with most configurations and go to least (so s+1, s-1, ..., 0)

    # best config out of each  batch is saved

    best_result = -1
    best_config = None

    for s in range(s_max, -1, -1):

        # calc starting number of configurations, (also Alg 1)
        n = math.ceil((B / R) * ((reduction_factor**s) / (s + 1)))


        # calc number of trees we start with in this bracket
        # is rounded to remove the float created with -s
        r = round(R * (reduction_factor**-s))


        config_list = []
        for stage in range(s + 1):
            # first bracket need to create te list and the configs
            if stage == 0:
                for config in range(n):
                    attempt_config = sample_configuration(rng)
                    eval_dict = evaluator(attempt_config, r, seed)
                    # nto save it, add bracket and stafe to it
                    eval_dict["bracket"] = s
                    eval_dict["stage"] = stage
                    all_results.append(eval_dict)
                    
                    # make info a tuple
                    attempt_tuple = (eval_dict["configuration"], eval_dict["objective"])
                    config_list.append(attempt_tuple)
            else:
                for index, (config, score) in enumerate(config_list):
                    eval_dict = evaluator(config, r, seed)

                    eval_dict["bracket"] = s
                    eval_dict["stage"] = stage
                    all_results.append(eval_dict)
                    
                    # change the values to the new score
                    config_list[index] = (eval_dict["configuration"], eval_dict["objective"])

            config_list = sorted(config_list, key=itemgetter(1), reverse=True) # reversed to make higher values be left

            if stage < s:
                # increasing amount of treess
                r = r * reduction_factor

                # remove configs deppending on the reduction_factor
                remaining_configs_amount = max(1, len(config_list) // reduction_factor)
                config_list = config_list[:remaining_configs_amount]

        # input in best config
        best_found_tuple = config_list[0]
        if  best_result < best_found_tuple[1]:
            best_result = best_found_tuple[1]
            best_config = best_found_tuple[0]

    return best_config, best_result # still need to do something with all results

            


            




















