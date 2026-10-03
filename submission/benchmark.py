"""LightGBM benchmark for the Credit Card Fraud Detection dataset."""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 16
DATASET_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")


def median_prediction_seconds(model, data, repeats):
    """Return median predict_proba wall time after an unmeasured warm-up."""
    model.predict_proba(data)
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict_proba(data)
        elapsed.append(time.perf_counter() - started)
    return float(np.median(elapsed))


def main():
    started = time.perf_counter()
    dataframe = pd.read_csv(DATASET_PATH)
    data_load_seconds = time.perf_counter() - started

    features = dataframe.drop(columns="Class")
    labels = dataframe["Class"]

    # 60% train, 20% validation, 20% test, stratified by the target.
    features_trainval, features_test, labels_trainval, labels_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )
    features_train, features_valid, labels_train, labels_valid = train_test_split(
        features_trainval,
        labels_trainval,
        test_size=0.25,
        random_state=SEED,
        stratify=labels_trainval,
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=2,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        features_train,
        labels_train,
        eval_set=[(features_valid, labels_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    training_seconds = time.perf_counter() - started

    probabilities = model.predict_proba(features_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    one_row = features_test.iloc[:1]
    batch = features_test.iloc[:1000]
    latency_repeats = 50
    batch_repeats = 10
    single_seconds = median_prediction_seconds(model, one_row, latency_repeats)
    batch_seconds = median_prediction_seconds(model, batch, batch_repeats)

    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": platform.machine(),
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset_rows": len(dataframe),
        "fraud_rows": int(labels.sum()),
        "seed": SEED,
        "split": {
            "train": len(features_train),
            "validation": len(features_valid),
            "test": len(features_test),
        },
        "n_jobs": 2,
        "decision_threshold": 0.5,
        "data_load_seconds": data_load_seconds,
        "training_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_),
        "auc_roc": float(roc_auc_score(labels_test, probabilities)),
        "accuracy": float(accuracy_score(labels_test, predictions)),
        "f1": float(f1_score(labels_test, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels_test, predictions, zero_division=0)),
        "latency_1_row_ms": single_seconds * 1000,
        "latency_repeats": latency_repeats,
        "batch_rows": len(batch),
        "batch_repeats": batch_repeats,
        "batch_1000_rows_seconds": batch_seconds,
        "throughput_1000_rows_per_second": len(batch) / batch_seconds,
        "timing_summary": "median; warm-up excluded; predict_proba on pandas input",
    }

    output = json.dumps(result, indent=2, allow_nan=False)
    RESULT_PATH.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
