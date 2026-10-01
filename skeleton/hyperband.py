"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any

from random_forest import Config, Evaluator

import math


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
    
    # amount of brackets is s_max + 1 (so range is s_max)
    R = max_trees/ min_trees
    s_max = math.floor(math.log(x=R, base=reduction_factor))
    B = (s_max + 1) * R
    #  we start with most configurations and go to least (so s+1, s-1, ..., 0)
    for s in range(s_max, -1, -1):

        # calc starting number of configurations, (also Alg 1)
        n = math.ceil((B / R) * ((reduction_factor**s) / (s + 1)))

        # calc number of trees we start with in this bracket
        r = R * (reduction_factor**-s)


