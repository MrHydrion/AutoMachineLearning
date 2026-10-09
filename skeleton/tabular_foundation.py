"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations
from typing import Any
from data_loading import DataSplits
from tabpfn import TabPFNClassifier
from time import perf_counter

import pandas as pd
import numpy as np

# prediction metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split



def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """

    # timer of the whole process
    start = perf_counter()

    # Combine training and validation data
    X_train = pd.concat([splits.X_train, splits.X_valid], ignore_index=True)
    y_train = np.concatenate([splits.y_train, splits.y_valid])

    # reducing the training size since it will otherwise be too big
    if len(X_train) > 10000:
        X_subset, _, y_subset, _ = train_test_split(
                X_train,
                y_train,
                train_size=10000,
                random_state=seed,
                stratify=y_train,
        )
    else:
        X_subset, y_subset = X_train, y_train

    # used TabPFN as the pre-trained tabular foundation model
    model = TabPFNClassifier(random_state=seed)
    print("Starting Fitting")
    fit_timer = perf_counter()
    model.fit(X_subset, y_subset)
    print("Ended Fitting")
    print(perf_counter() - fit_timer)

    # Predict in batches since cant otherwise with 8 gb GPU
    batch_size = 2000
    predictions = []

    print("start predicting")
    for i in range(0, len(splits.X_test), batch_size):
        X_batch = splits.X_test.iloc[i:i + batch_size]
        y_batch = model.predict(X_batch)
        predictions.extend(y_batch)

        print(
            f"Predicted {len(predictions)}/{len(splits.X_test)} | "
            f"Runtime: {perf_counter() - start:.1f}s",
            flush=True
        )


    y_pred = predictions
    elapsed_sec = perf_counter() - start
    accuracy = accuracy_score(splits.y_test, y_pred)
    macro_f1 = f1_score(splits.y_test, y_pred, average="macro")

    return {
        "metrics": {
            "accuracy": accuracy,
            "macro-f1": macro_f1,
        },
        "elapsed_sec": elapsed_sec,
        "n_train_samples": len(y_subset)
    }


    
