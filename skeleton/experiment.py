"""Example runner for a shared split, forest search, and final evaluation.

Adapt this flow to your experimental design. Results stay in memory; choose how
to save them and record the settings needed to reproduce your study.
"""

# ------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import platform
import sys
import sklearn

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_loading import (DATASETS, load_and_split, prepare_data, prepare_final_data)
from hyperband import optimise_hyperband
from random_forest import (N_JOBS, SEARCH_SPACE, final_test_evaluation, make_evaluator)
from random_search import optimise_random_search
from smbo import optimise_smbo
from tabular_foundation import run_foundation_model



# ------------------------------------------------------------------------
# Profiles
# ------------------------------------------------------------------------


# Update this if the provided largest dataset is replaced.
FOUNDATION_DATASET = "covertype"

# n_trials is the example evaluation budget for each of Random Search and SMBO.
# Choose budgets and a Hyperband schedule that support your justified comparison.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_samples": 2_500,
        "min_trees": 3,
        "max_trees": 27,
        "n_trials": 9,
    },
    "course": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 81,
        "n_trials": 16,
    },
    "full": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 243,
        "n_trials": 24,
    },
}


# ------------------------------------------------------------------------
# Arguments
# ------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="breast-w", choices=[*DATASETS, "all"])
    parser.add_argument("--profile", default="smoke", choices=PROFILES)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["default", "random", "smbo", "hyperband"],
        choices=["default", "random", "smbo", "hyperband", "foundation"],
    )
    # parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--output", type=Path, default=Path("results.json"))

    return parser.parse_args()



# ------------------------------------------------------------------------
# JSON
# ------------------------------------------------------------------------



def make_json(value: Any) -> Any:
    """Convert NumPy/scikit-learn values into JSON-compatible values."""

    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): make_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json(item) for item in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def create_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Create metadata describing the experiment and software environment."""

    profile = PROFILES[args.profile]

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "datasets": list(DATASETS.keys()),
        "methods": args.methods,
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "cache_dir": str(args.cache_dir),
        "max_samples": profile["max_samples"],
        "min_trees": profile["min_trees"],
        "max_trees": profile["max_trees"],
        "n_trials": profile["n_trials"],
        "search_space": make_json(SEARCH_SPACE),
        "random_forest_n_jobs": N_JOBS,
        "foundation_dataset": FOUNDATION_DATASET,
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_dataset(name: str, args: argparse.Namespace, seed: int) -> list[dict[str, Any]]:
    """Return example in-memory results; their structure is yours to adapt."""

    if args.methods == ["foundation"] and name != FOUNDATION_DATASET:
        if args.dataset == "all":
            return []
        raise ValueError(f"Foundation-only runs require --dataset {FOUNDATION_DATASET}")

    profile = PROFILES[args.profile]
    print(f"\n{'=' * 80}\nDataset: {name} | seed: {seed}\n{'=' * 80}", flush=True)

    splits = load_and_split(name, args.cache_dir, profile["max_samples"], args.split_seed)
    print(
        f"Rows: train={len(splits.X_train)}, "
        f"validation={len(splits.X_valid)}, "
        f"test={len(splits.X_test)}",
        flush=True
        )

    results = []


    forest_methods = {"default", "random", "smbo", "hyperband"}.intersection(args.methods)
    if forest_methods:
        X_train, X_valid = prepare_data(splits)
        evaluator = make_evaluator(
            X_train, splits.y_train, X_valid, splits.y_valid
        )
        final_arrays = prepare_final_data(splits)
        min_trees = int(profile["min_trees"])
        max_trees = int(profile["max_trees"])
        n_trials = int(profile["n_trials"])

        for method in ("default", "random", "smbo", "hyperband"):
            if method not in forest_methods:
                continue
            print(f"\nRunning {method} on {name}, seed {seed}", flush=True)
            start = perf_counter()

            if method == "default":
                config = {}  # Library defaults, with the common tree count.

                baseline_result = evaluator(config, max_trees, seed)
                baseline_result["trial"] = 1
                baseline_result["selection_method"] = "baseline"

                history = [baseline_result]
                selected_score = baseline_result["objective"]

            elif method == "random":
            
                (config, selected_score, history) = optimise_random_search(
                    evaluator,
                    n_trials,
                    max_trees,
                    seed
                )
                for trial, result in enumerate(history, start=1):
                    result.setdefault("trial", trial)
                    result.setdefault("selection_method", "random")


            elif method == "smbo":

                config, history = optimise_smbo(evaluator, n_trials, max_trees, seed)
                selected_score = max(result["objective"] for result in history)

                for trial, result in enumerate(history, start=1):
                    result.setdefault("trial", trial)


            else:

                (config, selected_score, history) = optimise_hyperband(
                    evaluator,
                    min_trees,
                    max_trees,
                    seed
                )
                for evaluation_number, result in enumerate(history, start=1):
                    result["evaluation"] = evaluation_number

            search_seconds = perf_counter() - start
            final_start = perf_counter()
            final_result = final_test_evaluation(config, max_trees, seed, *final_arrays)
            final_seconds = perf_counter() - final_start

            print(
                f"{name} / {method} / seed {seed}: "
                f"validation={selected_score:.6f}, "
                f"test_accuracy={final_result['metrics']['accuracy']:.6f}, "
                f"test_macro_f1={final_result['metrics']['macro-f1']:.6f}, "
                f"search_seconds={search_seconds:.2f}",
                flush=True
            )

            results.append(
                {
                    "dataset": name,
                    "method": method,
                    "seed": seed,
                    "split_seed": args.split_seed,
                    "profile": args.profile,
                    "n_trials": n_trials,
                    "min_trees": min_trees,
                    "max_trees": max_trees,
                    "selected_configuration": config,
                    "selected_validation_objective": selected_score,
                    "history": history,
                    "search_seconds": search_seconds,
                    "final_result": final_result,
                    "final_evaluation_seconds": final_seconds
                }
            )


    if "foundation" in args.methods and name == FOUNDATION_DATASET:
        print(f"\nRunning foundation model on {name}, seed {seed}", flush=True)
        
        foundation_start = perf_counter()
        foundation_result = run_foundation_model(splits, seed=seed)
        foundation_seconds = perf_counter() - foundation_start

        print(
            f"{name} / foundation / seed {seed}: "
            f"accuracy={foundation_result['metrics']['accuracy']:.6f}, "
            f"macro_f1={foundation_result['metrics']['macro-f1']:.6f}, "
            f"seconds={foundation_seconds:.2f}",
            flush=True
        )

        results.append(
            {
                "dataset": name,
                "method": "foundation",
                "seed": seed,
                "split_seed": args.split_seed,
                "profile": args.profile,
                "result": foundation_result,
                "elapsed_seconds": foundation_seconds
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the selected examples; add result saving before the main study."""
    # TODO: save results in a format of your choice, along with the settings
    # needed to reproduce the run. Retain enough information for your plots
    # and tables. This example only prints final results; it saves no files.

    args = parse_args()
    names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    metadata = create_metadata(args)
    all_results: list[dict[str, Any]] = []
    experiment_start = perf_counter()

    total_runs = len(names) * len(args.seeds)
    run_number = 0

    for seed in args.seeds:

        for name in names:

            run_number += 1

            print(
                f"\nStarting experiment "
                f"{run_number}/{total_runs}: "
                f"{name}, seed={seed}",
                flush=True
            )

            results = run_dataset(name, args, seed)
            all_results.extend(results)

    total_seconds = perf_counter() - experiment_start

    output = {
        "metadata": metadata,
        "total_experiment_seconds": total_seconds,
        "results": make_json(all_results)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with args.output.open("w", encoding="utf-8") as file:

        json.dump(output, file, indent=2, allow_nan=False )

    print(
        f"\n{'=' * 80}\n"
        f"Experiment finished.\n"
        f"Total runtime: {total_seconds:.2f} seconds\n"
        f"Results saved to: {args.output.resolve()}\n"
        f"Number of result records: {len(all_results)}\n"
        f"{'=' * 80}",
        flush=True
    )



if __name__ == "__main__":
    main()
