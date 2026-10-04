"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations

from typing import Any

from data_loading import DataSplits

from tabpfn import TabPFNClassifier

from time import perf_counter

# prediction metrics
from sklearn.metrics import accuracy_score, f1_score


def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """

    # used TabPFN as the pre-trained tabular foundation model
    model = TabPFNClassifier()

    # timer
    start = perf_counter()

    #fitting the model on the train data
    model.fit(splits.X_train, splits.y_train) # !! could need to use a subset but decide this late

    y_pred = model.predict(splits.X_test)

    elapsed_sec = perf_counter() - start

    accuracy = accuracy_score(splits.y_test, y_pred)
    macro_f1 = f1_score(splits.y_test, y_pred, average="macro")

    return {
        "metrics": {
            "accuracy": accuracy,
            "macro-f1": macro_f1,
        },
        "elapsed_sec": elapsed_sec
    }


    
