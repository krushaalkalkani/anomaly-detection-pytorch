"""Train an IsolationForest baseline on HDFS_v1 event-count splits."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from src.data import load_splits, prepare_and_save
from src.evaluate import evaluate_and_save

SEED = 42


def iforest_to_labels(raw_pred: np.ndarray) -> np.ndarray:
    """Map IsolationForest output (-1 anomaly, 1 normal) to 1/0 labels."""
    return np.where(raw_pred == -1, 1, 0).astype(np.int8)


def train_isolation_forest(x_train: np.ndarray, seed: int = SEED) -> IsolationForest:
    """Fit IsolationForest on the training event-count matrix."""
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train)
    return model


def ensure_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load saved splits, or build them if .npz files are missing."""
    try:
        return load_splits()
    except FileNotFoundError:
        return prepare_and_save(seed=SEED)


def main() -> None:
    """Train on train split, score test split, and write results/baseline.json."""
    splits = ensure_splits()
    x_train, _y_train = splits["train"]
    x_test, y_test = splits["test"]

    model = train_isolation_forest(x_train, seed=SEED)
    y_pred = iforest_to_labels(model.predict(x_test))
    evaluate_and_save(y_test, y_pred)


if __name__ == "__main__":
    main()
