"""
Analysis and visualization of AutoML experiment results.
"""


from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Configuration
# ============================================================================


RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis")


ALGORITHMS = [
    "default",
    "random",
    "smbo",
    "hyperband"
]


OPTIMIZERS = [
    "random",
    "smbo",
    "hyperband"
]


DATASETS = [
    "breast-w",
    "credit-g",
    "phoneme",
    "electricity",
    "covertype"
]

SEEDS = [
    17,
    18,
    19
]

FOUNDATION_DATASET = "covertype"
FOUNDATION_METHOD = "tabpfn"


# ============================================================================
# Loading results
# ============================================================================


def load_results(results_dir):
    """Load the metrics needed for algorithm comparison plots."""

    rows = []

    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        result = data["result"]

        if result["method"] not in ALGORITHMS:
            continue

        rows.append({
            "dataset": result["dataset"],
            "method": result["method"],
            "search_seconds": result["search_seconds"],
            "accuracy": result["final_result"]["metrics"]["accuracy"]
        })

    return pd.DataFrame(rows)



def load_histories(results_dir):
    """Load the timing and objective history needed for validation-vs-time plots."""

    rows = []

    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        result = data["result"]
        method = result["method"]

        if method not in OPTIMIZERS:
            continue

        cumulative_seconds = 0.0

        for index, evaluation in enumerate(result.get("history", []), start=1):
            cumulative_seconds += float(evaluation.get("elapsed_sec", 0.0))

            rows.append({
                "dataset": result["dataset"],
                "method": method,
                "seed": result["seed"],
                "evaluation": index,
                "objective": evaluation.get("objective"),
                "cumulative_seconds": cumulative_seconds
            })

    return pd.DataFrame(rows)



# ============================================================================
# Plot helpers
# ============================================================================


def style_axis(ax):

    ax.grid(
        axis="y",
        which="major",
        linestyle="--",
        alpha=0.7,
    )

    ax.set_axisbelow(True)


def save_plot(fig, output_path):

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output_path}")



# ============================================================================
# algorithm comparison
# ============================================================================


def plot_algorithm_comparison_with_foundation(dataframe, foundation_dataframe):
    """
    Compare the forest methods with TabPFN.
    """

    output_dir = (OUTPUT_DIR/ "algorithm_comparison")
    metrics = {
        "accuracy": "Mean test accuracy",
        "search_seconds": "Mean computational effort (seconds)"
    }



    forest_grouped = (dataframe.groupby(["dataset", "method"], as_index=False,)
        .agg(
            {
                "search_seconds": [
                    "mean",
                    "min",
                    "max",
                ],
                "accuracy": [
                    "mean",
                    "min",
                    "max",
                ],
            }
        )
    )

    forest_grouped.columns = [
        "_".join(column).strip("_")
        if isinstance(column, tuple)
        else column
        for column in forest_grouped.columns
    ]


    if not foundation_dataframe.empty:

        tabpfn_data = foundation_dataframe[
            foundation_dataframe["dataset"] == FOUNDATION_DATASET
        ]

        tabpfn_rows = []

        if not tabpfn_data.empty:

            tabpfn_rows.append(
                {
                    "dataset": FOUNDATION_DATASET,
                    "method": FOUNDATION_METHOD,
                    "accuracy_mean": (tabpfn_data["accuracy"].mean()),
                    "accuracy_min": (tabpfn_data["accuracy"].min()),
                    "accuracy_max": (tabpfn_data["accuracy"].max()),
                    "search_seconds_mean": (tabpfn_data["elapsed_seconds"].mean()),
                    "search_seconds_min": (tabpfn_data["elapsed_seconds"].min()),
                    "search_seconds_max": (tabpfn_data["elapsed_seconds"].max())
                }
            )

        if tabpfn_rows:

            tabpfn_grouped = pd.DataFrame(tabpfn_rows)
            forest_grouped = pd.concat([forest_grouped, tabpfn_grouped], ignore_index=True)


    forest_methods = ALGORITHMS
    bar_width = 0.16

    offsets = {
        "default": -1.5 * bar_width,
        "random": -0.5 * bar_width,
        "smbo": 0.5 * bar_width,
        "hyperband": 1.5 * bar_width,
        "tabpfn": 2.5 * bar_width
    }

    dataset_positions = range(len(DATASETS))

    for metric, ylabel in metrics.items():

        fig, ax = plt.subplots(figsize=(11, 6))

        for method in forest_methods:

            method_data = (
                forest_grouped[forest_grouped["method"] == method]
                .set_index("dataset")
                .reindex(DATASETS)
            )

            positions = [position + offsets[method] for position in dataset_positions]
            means = method_data[f"{metric}_mean"].to_numpy()
            minimums = method_data[f"{metric}_min"].to_numpy()
            maximums = method_data[f"{metric}_max"].to_numpy()
            lower_errors = (means - minimums)
            upper_errors = (maximums - means)

            ax.bar(
                positions,
                means,
                width=bar_width,
                label=method.capitalize(),
                yerr=[lower_errors, upper_errors],
                capsize=5
            )

        tabpfn_data = forest_grouped[forest_grouped["method"] == FOUNDATION_METHOD]

        if not tabpfn_data.empty:

            tabpfn_row = tabpfn_data[tabpfn_data["dataset"] == FOUNDATION_DATASET]

            if not tabpfn_row.empty:

                covertype_position = (DATASETS.index(FOUNDATION_DATASET))
                position = (covertype_position + offsets[FOUNDATION_METHOD])
                mean = tabpfn_row[f"{metric}_mean"].iloc[0]
                minimum = tabpfn_row[f"{metric}_min"].iloc[0]
                maximum = tabpfn_row[f"{metric}_max"].iloc[0]

                ax.bar(
                    position,
                    mean,
                    width=bar_width,
                    label="TabPFN",
                    yerr=[[mean - minimum], [maximum - mean]],
                    capsize=5
                )


        if metric == "search_seconds":
            ax.set_yscale("log")

        if metric == "accuracy":
            ax.set_ylim(0.7, 1.0)

        ax.set_title(f"Algorithm comparison including TabPFN — {ylabel}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(ylabel)
        style_axis(ax)
        ax.set_xticks(list(dataset_positions))
        ax.set_xticklabels(DATASETS)
        ax.legend()
        output_path = (output_dir / f"algorithm_comparison_with_tabpfn_{metric}.png")
        save_plot(fig, output_path)




# ============================================================================
# best validation accuracy vs cumulative search time
# ============================================================================



def plot_validation_vs_time(histories):
    """
    Plot mean best-so-far validation accuracy against cumulative search time.
    Each optimizer is averaged across all available seeds.
    """

    output_dir = (OUTPUT_DIR / "optimization" / "validation_vs_time")

    for dataset in DATASETS:

        dataset_data = histories[histories["dataset"] == dataset].copy()

        if dataset_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 6))

        for method in OPTIMIZERS:

            method_data = dataset_data[dataset_data["method"] == method]

            seed_curves = []

            for seed in SEEDS:

                seed_data = method_data[method_data["seed"] == seed].sort_values("cumulative_seconds")

                if seed_data.empty:
                    continue

                seed_data = seed_data.copy()
                seed_data["best_so_far"] = (seed_data["objective"].cummax())
                seed_data = seed_data.dropna(subset=["cumulative_seconds", "best_so_far"])

                if seed_data.empty:
                    continue

                seed_curves.append(
                    (
                        seed_data["cumulative_seconds"].to_numpy(),
                        seed_data["best_so_far"].to_numpy(),
                    )
                )

            if not seed_curves:
                continue

            common_end_time = min(times[-1] for times, _ in seed_curves)
            common_start_time = max(times[0] for times, _ in seed_curves)

            if common_start_time >= common_end_time:
                continue

            time_points = sorted({
                float(time)
                for times, _ in seed_curves
                for time in times
                if common_start_time <= time <= common_end_time
            })

            if not time_points:
                continue

            interpolated_curves = []

            for times, accuracies in seed_curves:

                interpolated_accuracy = pd.Series(index=time_points, data=np.interp(time_points, times, accuracies))
                interpolated_curves.append(interpolated_accuracy.to_numpy())

            mean_accuracy = pd.DataFrame(interpolated_curves).mean(axis=0).to_numpy()

            ax.plot(
                time_points,
                mean_accuracy,
                linewidth=2,
                label=method.capitalize()
            )

        ax.set_title(f"Mean optimization efficiency — {dataset}")
        ax.set_xlabel("Cumulative search time (seconds)")
        ax.set_ylabel("Mean best-so-far validation accuracy")
        style_axis(ax)
        ax.legend()
        output_path = (output_dir / f"{dataset}_validation_vs_time.png")
        save_plot(fig, output_path)


def load_foundation_results(results_dir):
    """
    Load TabPFN results from the foundation-model JSON files.
    """

    rows = []

    foundation_files = sorted(results_dir.glob(f"{FOUNDATION_DATASET}_foundation_seed_*.json"))

    print(f"Found {len(foundation_files)} foundation-model result files.")

    for path in foundation_files:

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        result = data["result"]
        foundation_result = result["result"]

        rows.append(
            {
                "dataset": result["dataset"],
                "method": FOUNDATION_METHOD,
                "seed": result["seed"],
                "accuracy": foundation_result["metrics"]["accuracy"],
                "macro_f1": foundation_result["metrics"]["macro-f1"],
                "elapsed_seconds": result["elapsed_seconds"]
            }
        )

    return pd.DataFrame(rows)




def main():
    print("=" * 80)
    print("Loading experiment results")
    print("=" * 80)

    dataframe = load_results(RESULTS_DIR)
    foundation_dataframe = load_foundation_results(RESULTS_DIR)
    histories = load_histories(RESULTS_DIR)

    print(f"Loaded {len(dataframe)} algorithm results.")
    print(f"Loaded {len(histories)} optimizer evaluations.")

    print("\n" + "=" * 80)
    print("Creating algorithm comparison plots")
    print("=" * 80)

    plot_algorithm_comparison_with_foundation(dataframe, foundation_dataframe)

    print("\n" + "=" * 80)
    print("Creating validation-vs-time plots")
    print("=" * 80)

    plot_validation_vs_time(histories)

    print("\n" + "=" * 80)
    print("Analysis finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()

