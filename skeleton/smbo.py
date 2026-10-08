"""Optional SMBO interface for the shared forest search space.

Implement the method, connect a suitable package, or replace this interface.
Choose and explain the method's settings and how you retain search results.
"""

from __future__ import annotations
from typing import Any
from random_forest import Config, Evaluator, SEARCH_SPACE
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


# generate every configuration in the search space
def generate_configurations() -> list[Config]:

    parameter_names = list(SEARCH_SPACE.keys())
    parameter_values = [SEARCH_SPACE[name] for name in parameter_names]

    configurations = []

    for values in product(*parameter_values):
        config = dict(zip(parameter_names, values))
        configurations.append(config)

    return configurations


# randomly select unique configurations for the initial design
def select_initial_configurations(
    configurations: list[Config], 
    n_initial: int, 
    rng: np.random.Generator
) -> list[Config]:

    indices = rng.choice(len(configurations), size=n_initial, replace=False)

    return [configurations[int(index)] for index in indices]


# evaluate each configuration and return the evaluation history
def evaluate_configurations(
    configurations: list[Config],
    evaluator: Evaluator,
    n_trees: int,
    seed: int
) -> list[dict[str, Any]]:

    history = []

    for config in configurations:
        result = evaluator(config, n_trees, seed)
        history.append(result)

    return history


# convert categorical hyperparameter configurations into numeric features
def encode_configurations(
    configurations: list[Config],
    encoder: OneHotEncoder
) -> np.ndarray:

    rows = []

    for config in configurations:
        rows.append({
            "max_depth": str(config["max_depth"]),
            "max_features": str(config["max_features"]),
            "min_samples_leaf": str(config["min_samples_leaf"]),
            "min_samples_split": str(config["min_samples_split"])
        })

    dataframe = pd.DataFrame(rows)

    return encoder.transform(dataframe)


# fit an encoder on every configuration in the search space
def create_encoder(
    configurations: list[Config]
) -> OneHotEncoder:

    rows = []

    for config in configurations:
        rows.append({
            "max_depth": str(config["max_depth"]),
            "max_features": str(config["max_features"]),
            "min_samples_leaf": str(config["min_samples_leaf"]),
            "min_samples_split": str(config["min_samples_split"])
        })

    dataframe = pd.DataFrame(rows)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(dataframe)

    return encoder


# fit a surrogate model to the observed configuration performances
def fit_surrogate(
    history: list[dict[str, Any]],
    encoder: OneHotEncoder,
    n_estimators: int = 100,
    seed: int = 17
) -> RandomForestRegressor:

    configurations = [result["configuration"] for result in history]
    objectives = np.array([result["objective"] for result in history])

    X = encode_configurations(configurations, encoder)

    surrogate = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    surrogate.fit(X, objectives)

    return surrogate


# select the next configuration using an Upper Confidence Bound
def select_next_configuration(
    surrogate: RandomForestRegressor,
    candidate_configurations: list[Config],
    encoder: OneHotEncoder,
    beta: float = 1.0
) -> tuple[Config, float, float, float]:

    X_candidates = encode_configurations(candidate_configurations, encoder)
    tree_predictions = np.array([tree.predict(X_candidates) for tree in surrogate.estimators_])

    predicted_mean = tree_predictions.mean(axis=0)
    predicted_std = tree_predictions.std(axis=0)

    acquisition = predicted_mean + beta * predicted_std
    best_index = int(np.argmax(acquisition))

    return (
        candidate_configurations[best_index],
        float(predicted_mean[best_index]),
        float(predicted_std[best_index]),
        float(acquisition[best_index])
    )


def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: use SMBO to choose configurations based on previous evaluations.

    Use the shared search space and train each forest with n_trees trees.
    Use up to n_trials evaluations, including any initial evaluations.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """

    rng = np.random.default_rng(seed)
    configurations = generate_configurations()
    n_initial = min(5, max(2, n_trials // 3))
    encoder = create_encoder(configurations)

    initial_configurations = select_initial_configurations(configurations, n_initial, rng)
    history = evaluate_configurations(initial_configurations, evaluator, n_trees, seed)

    for trial, result in enumerate(history, start=1):
        result["trial"] = trial
        result["selection_method"] = "initial"

    evaluated = {tuple(result["configuration"][name] for name in SEARCH_SPACE) for result in history}

    while len(history) < n_trials:

        remaining_configurations = [
            config
            for config in configurations
            if tuple(config[name] for name in SEARCH_SPACE) not in evaluated
        ]

        surrogate = fit_surrogate(history, encoder, seed=seed)

        (next_config, predicted_mean, predicted_std, acquisition) = select_next_configuration(
            surrogate,
            remaining_configurations,
            encoder,
            beta=1.0
        )

        result = evaluator(next_config, n_trees, seed)

        result["trial"] = len(history) + 1
        result["selection_method"] = "ucb"
        result["predicted_mean"] = predicted_mean
        result["predicted_std"] = predicted_std
        result["acquisition"] = acquisition

        history.append(result)
        evaluated.add(tuple(next_config[name] for name in SEARCH_SPACE))

    best_result = max(history, key=lambda result: result["objective"])
    best_config = best_result["configuration"]

    return best_config, history



