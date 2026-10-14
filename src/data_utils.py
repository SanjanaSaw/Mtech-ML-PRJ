"""Dataset loading and leakage-aware chronological graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class TemporalSplits:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor


@dataclass
class EllipticData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    timestep: torch.Tensor
    tx_id: np.ndarray
    feature_names: list[str]

    @property
    def labeled_mask(self) -> torch.Tensor:
        return self.y >= 0


def _map_elliptic_labels(labels: Iterable[object]) -> np.ndarray:
    label_map = {"unknown": -1, "1": 1, "2": 0}
    normalized = [str(value).strip().lower() for value in labels]
    unexpected = sorted(set(normalized) - set(label_map))
    if unexpected:
        raise ValueError(f"Unexpected Elliptic labels: {unexpected}.")
    return np.asarray([label_map[value] for value in normalized], dtype=np.int64)


def _map_pyg_labels(labels: np.ndarray) -> np.ndarray:
    """Convert PyG's 0=licit, 1=illicit, 2=unknown convention."""
    labels = np.asarray(labels, dtype=np.int64)
    if np.all(np.isin(labels, [0, 1, 2])):
        return np.where(labels == 2, -1, labels)
    return labels


def validate_elliptic(data: EllipticData) -> dict[str, int]:
    """Validate graph, metadata, labels, and feature dimensions before training."""
    node_count = data.x.shape[0]
    if data.x.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {tuple(data.x.shape)}.")
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index with shape [2, E], got {tuple(data.edge_index.shape)}."
        )
    if data.y.shape[0] != node_count or data.timestep.shape[0] != node_count:
        raise ValueError("Node features, labels, and timesteps are not aligned.")
    if len(data.feature_names) != data.x.shape[1]:
        raise ValueError("Feature names do not match the feature matrix width.")
    if len(data.tx_id) != node_count or len(set(map(str, data.tx_id))) != node_count:
        raise ValueError("Transaction IDs are missing, duplicated, or misaligned.")
    if data.edge_index.numel():
        if int(data.edge_index.min()) < 0 or int(data.edge_index.max()) >= node_count:
            raise ValueError("The edge list contains an endpoint outside the node range.")
    if not bool(torch.isfinite(data.x).all()):
        raise ValueError("Node features contain NaN or infinite values.")
    if not bool(torch.isin(data.y, torch.tensor([-1, 0, 1])).all()):
        raise ValueError("Labels must use -1=unknown, 0=licit, and 1=illicit.")
    return {
        "nodes": int(node_count),
        "edges": int(data.edge_index.shape[1]),
        "features": int(data.x.shape[1]),
        "timesteps": int(data.timestep.max()),
        "unknown": int((data.y == -1).sum()),
        "licit": int((data.y == 0).sum()),
        "illicit": int((data.y == 1).sum()),
    }


def _read_raw_metadata(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features_path = root / "raw" / "elliptic_txs_features.csv"
    classes_path = root / "raw" / "elliptic_txs_classes.csv"
    if not features_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"Expected Elliptic raw files under {root / 'raw'}."
        )
    import pandas as pd

    features = pd.read_csv(features_path, header=None)
    classes = pd.read_csv(classes_path)
    if classes.shape[1] < 2:
        classes = pd.read_csv(classes_path, header=None)
    if classes.shape[1] != 2:
        raise ValueError(
            f"Expected transaction ID and label columns in {classes_path}, "
            f"found {classes.shape[1]} columns."
        )
    classes.columns = ["tx_id", "label"]
    tx_id = features.iloc[:, 0].to_numpy()
    timestep = features.iloc[:, 1].astype(int).to_numpy()
    labels = classes.set_index(classes.columns[0]).reindex(tx_id).iloc[:, 0]
    if labels.isna().any():
        missing = int(labels.isna().sum())
        raise ValueError(f"{missing} feature transaction IDs have no class row.")
    y = _map_elliptic_labels(labels)
    feature_names = [f"feature_{i}" for i in range(1, features.shape[1] - 1)]
    return tx_id, timestep, y, feature_names


def load_elliptic(root: str | Path = "data/elliptic") -> EllipticData:
    """Load the existing processed PyG graph and align raw timestep/labels by tx_id.

    The raw feature file stores tx_id and timestep before the 165 numeric
    attributes. Those metadata columns are not passed to the neural encoders.
    """
    root = Path(root)
    processed = root / "processed" / "data.pt"
    if not processed.exists():
        raise FileNotFoundError(f"Processed graph not found: {processed}")
    obj = torch.load(processed, map_location="cpu", weights_only=False)
    data = obj[0] if isinstance(obj, (list, tuple)) else obj
    if isinstance(data, dict):
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError(
                "torch-geometric is required to load the processed Elliptic graph."
            ) from exc
        data = Data(**data)
    tx_id, timestep, labels, feature_names = _read_raw_metadata(root)
    if data.x.shape[0] != len(tx_id):
        raise ValueError(
            f"Node/metadata mismatch: graph has {data.x.shape[0]} nodes, "
            f"raw features have {len(tx_id)} rows."
        )
    graph_tx_id = getattr(data, "tx_id", None)
    if graph_tx_id is not None:
        graph_tx_id = np.asarray(graph_tx_id).astype(str)
        if not np.array_equal(graph_tx_id, tx_id.astype(str)):
            raise ValueError(
                "Processed graph transaction order does not match the raw feature rows."
            )
    # The standard Elliptic PyG object has 0=licit, 1=illicit, 2=unknown.
    # Check its labels against the raw ID-aligned class file before using either.
    graph_y = data.y.detach().cpu().numpy().astype(np.int64)
    aligned_graph_y = _map_pyg_labels(graph_y)
    if not np.array_equal(aligned_graph_y, labels):
        raise ValueError(
            "Processed graph labels do not align with raw labels in transaction order."
        )
    result = EllipticData(
        x=data.x.detach().cpu().float(),
        edge_index=data.edge_index.detach().cpu().long(),
        y=torch.from_numpy(labels),
        timestep=torch.tensor(timestep, dtype=torch.long),
        tx_id=tx_id,
        feature_names=feature_names,
    )
    validate_elliptic(result)
    return result


def make_temporal_splits(
    timestep: torch.Tensor,
    train_steps: Iterable[int] = range(1, 25),
    validation_steps: Iterable[int] = range(25, 35),
    test_steps: Iterable[int] = range(35, 50),
) -> TemporalSplits:
    """Return disjoint masks for timesteps 1-24, 25-34, and 35-49 by default."""
    train_steps = list(train_steps)
    validation_steps = list(validation_steps)
    test_steps = list(test_steps)
    if (
        set(train_steps) & set(validation_steps)
        or set(train_steps) & set(test_steps)
        or set(validation_steps) & set(test_steps)
    ):
        raise ValueError("Training, validation, and test timesteps must be disjoint.")
    train = torch.isin(timestep, torch.tensor(train_steps, dtype=timestep.dtype))
    validation = torch.isin(
        timestep, torch.tensor(validation_steps, dtype=timestep.dtype)
    )
    test = torch.isin(timestep, torch.tensor(test_steps, dtype=timestep.dtype))
    return TemporalSplits(train=train, validation=validation, test=test)


def filter_edges_by_time(
    edge_index: torch.Tensor,
    timestep: torch.Tensor,
    max_timestep: int,
    include_cross_boundary: bool = False,
) -> torch.Tensor:
    """Filter message passing edges to information available by ``max_timestep``.

    With ``include_cross_boundary=False`` both endpoints must be observed by the
    cutoff. This is the strict inductive graph used for future-node evaluation.
    """
    src, dst = edge_index
    if include_cross_boundary:
        keep = (timestep[src] <= max_timestep) | (timestep[dst] <= max_timestep)
    else:
        keep = (timestep[src] <= max_timestep) & (timestep[dst] <= max_timestep)
    return edge_index[:, keep]


def node_degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return undirected degree, avoiding the old outgoing-only calculation."""
    src, dst = edge_index
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, src, torch.ones_like(src))
    degree.scatter_add_(0, dst, torch.ones_like(dst))
    return degree
