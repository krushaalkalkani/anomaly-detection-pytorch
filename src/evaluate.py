"""Compute and persist classification metrics for anomaly detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return precision, recall, F1, and confusion matrix for binary labels."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print metric values to stdout."""
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"confusion_matrix (rows=true [0,1], cols=pred [0,1]):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")


def save_metrics(metrics: dict, path: Path) -> None:
    """Write metrics dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def evaluate_and_save(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path | None = None,
) -> dict:
    """Compute metrics, print them, and save JSON under results/."""
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)
    destination = out_path or (RESULTS_DIR / "baseline.json")
    save_metrics(metrics, destination)
    print(f"saved: {destination}")
    return metrics
