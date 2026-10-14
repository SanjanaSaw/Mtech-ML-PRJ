import numpy as np
import pytest
import torch

from src.data_utils import (
    EllipticData,
    _map_elliptic_labels,
    _map_pyg_labels,
    filter_edges_by_time,
    make_temporal_splits,
    validate_elliptic,
)


def test_elliptic_labels_map_to_binary_and_unknown_is_excluded():
    assert _map_elliptic_labels(["unknown", "1", "2", 1, 2]).tolist() == [
        -1, 1, 0, 1, 0
    ]


def test_elliptic_label_mapping_rejects_unrecognized_values():
    with pytest.raises(ValueError, match="Unexpected Elliptic labels"):
        _map_elliptic_labels(["unknown", "not-a-class"])


def test_pyg_dataset_label_mapping_matches_official_convention():
    assert _map_pyg_labels(np.array([0, 1, 2])).tolist() == [0, 1, -1]
    assert _map_pyg_labels(np.array([-1, 0, 1])).tolist() == [-1, 0, 1]


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


def test_make_temporal_splits_defaults_to_requested_periods():
    timestep = torch.arange(1, 50)
    splits = make_temporal_splits(timestep)

    assert torch.all(timestep[splits.train] <= 24)
    assert torch.equal(timestep[splits.train], torch.arange(1, 25))
    assert torch.equal(timestep[splits.validation], torch.arange(25, 35))
    assert torch.equal(timestep[splits.test], torch.arange(35, 50))
    assert not (splits.train & splits.validation).any()
    assert not (splits.train & splits.test).any()
    assert not (splits.validation & splits.test).any()


def test_make_temporal_splits_rejects_overlapping_periods():
    with pytest.raises(ValueError, match="disjoint"):
        make_temporal_splits(
            torch.tensor([1, 2, 3]),
            train_steps=[1, 2],
            validation_steps=[2],
            test_steps=[3],
        )


def test_filter_edges_by_time_distinguishes_strict_and_cross_boundary_modes():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]])
    timestep = torch.tensor([1, 3, 2, 4])

    strict = filter_edges_by_time(edge_index, timestep, max_timestep=2)
    cross_boundary = filter_edges_by_time(
        edge_index, timestep, max_timestep=2, include_cross_boundary=True
    )

    assert strict.tolist() == [[0], [2]]
    assert cross_boundary.tolist() == [[0, 0, 1, 2], [1, 2, 2, 3]]


def test_causal_edges_never_include_future_nodes_and_expand_monotonically():
    edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 4]])
    timestep = torch.tensor([1, 2, 2, 3, 4])

    by_cutoff = [
        filter_edges_by_time(edge_index, timestep, max_timestep=cutoff)
        for cutoff in (1, 2, 3, 4)
    ]
    for cutoff, edges in enumerate(by_cutoff, start=1):
        assert torch.all(timestep[edges] <= cutoff)
    assert all(
        set(map(tuple, previous.T.tolist())).issubset(
            set(map(tuple, current.T.tolist()))
        )
        for previous, current in zip(by_cutoff, by_cutoff[1:])
    )


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
