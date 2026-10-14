import numpy as np
import pytest

from src.evaluation import binary_metrics, select_threshold


def test_select_threshold_handles_empty_and_single_class_labels():
    assert select_threshold(np.array([]), np.array([])) == 0.5
    assert select_threshold(np.array([0, 0]), np.array([0.1, 0.9])) == 0.5
    assert select_threshold(np.array([1, 1]), np.array([0.1, 0.9])) == 0.5


def test_select_threshold_chooses_threshold_with_best_f1():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])

    assert select_threshold(y_true, probability) == pytest.approx(0.6)


def test_binary_metrics_reports_confusion_counts_and_rates():
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.2, 0.7, 0.6, 0.1]),
        threshold=0.5,
    )

    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TP"] == 1
    assert metrics["Precision"] == pytest.approx(0.5)
    assert metrics["Recall"] == pytest.approx(0.5)
    assert metrics["F1"] == pytest.approx(0.5)
    assert metrics["False Positive Rate"] == pytest.approx(0.5)
    assert metrics["threshold"] == 0.5


@pytest.mark.parametrize(
    "labels, expected_recall, expected_precision",
    [
        (np.array([0, 0]), 0.0, 0.0),
        (np.array([1, 1]), 0.5, 1.0),
    ],
)
def test_binary_metrics_handles_single_class_labels(labels, expected_recall, expected_precision):
    metrics = binary_metrics(labels, np.array([0.2, 0.8]), threshold=0.5)

    assert np.isnan(metrics["ROC-AUC"])
    assert metrics["Recall"] == expected_recall
    assert metrics["Precision"] == expected_precision
    assert metrics["MCC"] == 0.0
    assert np.isnan(metrics["PR-AUC"])


def test_threshold_selection_rejects_misaligned_or_nonbinary_validation_data():
    with pytest.raises(ValueError, match="same shape"):
        select_threshold(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="only 0"):
        select_threshold(np.array([-1, 1]), np.array([0.2, 0.8]))


def test_temporal_metrics_includes_counts_and_illicit_proportions():
    from src.evaluation import temporal_metrics

    result = temporal_metrics(
        np.array([0, 1, 1]),
        np.array([0.1, 0.8, 0.4]),
        np.array([35, 35, 36]),
        threshold=0.5,
    )

    assert result["transaction_count"].tolist() == [2, 1]
    assert result["illicit_count"].tolist() == [1, 1]
    assert result["illicit_proportion"].tolist() == pytest.approx([0.5, 1.0])
    assert np.isnan(result.loc[1, "ROC-AUC"])
