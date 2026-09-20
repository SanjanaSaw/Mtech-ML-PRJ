"""Leakage-aware metrics, threshold selection, and result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability)
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "ROC-AUC": roc_auc_score(y_true, probability) if len(np.unique(y_true)) > 1 else np.nan,
        "PR-AUC": average_precision_score(y_true, probability) if y_true.sum() else np.nan,
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Recall": recall_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_true, predicted) if len(np.unique(predicted)) > 1 else 0.0,
        "Balanced Accuracy": balanced_accuracy_score(y_true, predicted),
        "False Positive Rate": fp / max(fp + tn, 1),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "threshold": threshold,
    }


def temporal_metrics(
    y_true: np.ndarray, probability: np.ndarray, timesteps: np.ndarray, threshold: float
) -> pd.DataFrame:
    rows = []
    for step in sorted(np.unique(timesteps)):
        mask = timesteps == step
        if not mask.any():
            continue
        row = binary_metrics(y_true[mask], probability[mask], threshold)
        row["timestep"] = int(step)
        row["illicit_count"] = int(y_true[mask].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_predictions(
    path: str | Path,
    tx_id: np.ndarray,
    timestep: np.ndarray,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "tx_id": tx_id, "timestep": timestep, "y_true": y_true,
        "probability": probability, "prediction": probability >= threshold,
    }).to_csv(path, index=False)

