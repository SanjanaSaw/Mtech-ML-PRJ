import numpy as np
import pytest
import torch

from src.data_utils import (
    EllipticData,
    filter_edges_by_time,
    make_temporal_splits,
    validate_elliptic,
)


def _data(**overrides):
    values = dict(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]]),
        y=torch.tensor([-1, 0, 1, 1]),
        timestep=torch.tensor([1, 1, 2, 3]),
        tx_id=np.array(["a", "b", "c", "d"]),
        feature_names=["f0", "f1"],
    )
    values.update(overrides)
    return EllipticData(**values)


def test_make_temporal_splits_creates_chronological_masks():
    splits = make_temporal_splits(
        torch.tensor([1, 2, 3, 4, 5]),
        train_steps=[1, 2],
        validation_steps=[3],
        test_steps=[4, 5],
    )

    assert splits.train.tolist() == [True, True, False, False, False]
    assert splits.validation.tolist() == [False, False, True, False, False]
    assert splits.test.tolist() == [False, False, False, True, True]


def test_make_temporal_splits_supports_empty_and_generator_step_lists():
    splits = make_temporal_splits(
        torch.tensor([1, 2]),
        train_steps=iter(()),
        validation_steps=iter((2,)),
        test_steps=iter(()),
    )

    assert not splits.train.any()
    assert splits.validation.tolist() == [False, True]
    assert not splits.test.any()


def test_filter_edges_by_time_distinguishes_strict_and_cross_boundary_modes():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    timestep = torch.tensor([1, 3, 2, 4])

    strict = filter_edges_by_time(edge_index, timestep, max_timestep=2)
    cross_boundary = filter_edges_by_time(
        edge_index, timestep, max_timestep=2, include_cross_boundary=True
    )

    assert strict.tolist() == [[0], [2]]
    assert cross_boundary.tolist() == [[0, 0, 1, 2], [1, 2, 2, 3]]


def test_validate_elliptic_returns_dataset_summary():
    assert validate_elliptic(_data()) == {
        "nodes": 4,
        "edges": 4,
        "features": 2,
        "timesteps": 3,
        "unknown": 1,
        "licit": 1,
        "illicit": 2,
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"edge_index": torch.tensor([0, 1, 2, 3])}, "edge_index"),
        ({"y": torch.tensor([-1, 0, 1])}, "aligned"),
        ({"tx_id": np.array(["a", "b", "b", "d"])}, "Transaction IDs"),
        ({"edge_index": torch.tensor([[0], [4]])}, "endpoint"),
        ({"x": torch.tensor([[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])}, "NaN"),
        ({"y": torch.tensor([-1, 0, 1, 2])}, "Labels"),
    ],
)
def test_validate_elliptic_rejects_malformed_data(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_elliptic(_data(**overrides))
