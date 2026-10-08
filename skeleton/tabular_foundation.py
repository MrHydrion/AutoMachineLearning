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
from sklearn.model_selection import train_test_split



def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """
    # reducing the training size since it will otherwise be too big
    X_subset, _, y_subset, _ = train_test_split(
        splits.X_train,
        splits.y_train,
        train_size=10000,
        random_state=seed,
        stratify=splits.y_train,
    )

    # used TabPFN as the pre-trained tabular foundation model
    model = TabPFNClassifier(random_state=seed)

    # timer
    start = perf_counter()

    #fitting the model on the train data
    model.fit(X_subset, y_subset) # !! could need to use a subset but decide this late

    # Predict in batches to prevent CUDA out-of-memory errors
    batch_size = 2000
    predictions = []

    print("start predicting")
    for i in range(0, len(splits.X_test), batch_size):
        X_batch = splits.X_test[i:i + batch_size]
        y_batch = model.predict(X_batch)
        predictions.extend(y_batch)
        print(i)

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


    
