"""Leakage-aware metrics, threshold selection, and result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
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
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability)
    if y_true.shape != probability.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite.")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0=licit and 1=illicit.")
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability)
    if y_true.shape != probability.shape:
        raise ValueError("Labels and probabilities must have the same shape.")
    if not np.isfinite(probability).all():
        raise ValueError("Probabilities must be finite.")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only 0=licit and 1=illicit.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability in [0, 1].")
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "ROC-AUC": roc_auc_score(y_true, probability) if both_classes else np.nan,
        "PR-AUC": average_precision_score(y_true, probability) if both_classes else np.nan,
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
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    timesteps = np.asarray(timesteps)
    if y_true.shape != probability.shape or y_true.shape != timesteps.shape:
        raise ValueError("Labels, probabilities, and timesteps must have the same shape.")
    rows = []
    for step in sorted(np.unique(timesteps)):
        mask = timesteps == step
        if not mask.any():
            continue
        row = binary_metrics(y_true[mask], probability[mask], threshold)
        row["timestep"] = int(step)
        row["transaction_count"] = int(mask.sum())
        row["illicit_count"] = int(y_true[mask].sum())
        row["illicit_proportion"] = float(y_true[mask].mean())
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
