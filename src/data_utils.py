"""Dataset loading and leakage-aware chronological graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
    classes.columns = ["tx_id", "label"][: classes.shape[1]]
    tx_id = features.iloc[:, 0].to_numpy()
    timestep = features.iloc[:, 1].astype(int).to_numpy()
    labels = classes.set_index(classes.columns[0]).reindex(tx_id).iloc[:, 0]
    # PyG represents unknown/licit/illicit as 0/1/2. Internally use -1/0/1.
    label_map = {"unknown": -1, "1": 1, "2": 0, 1: 1, 2: 0}
    y = labels.map(lambda value: label_map.get(value, -1)).to_numpy(dtype=np.int64)
    feature_names = [f"feature_{i}" for i in range(features.shape[1] - 2)]
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
    tx_id, timestep, labels, feature_names = _read_raw_metadata(root)
    if data.x.shape[0] != len(tx_id):
        raise ValueError(
            f"Node/metadata mismatch: graph has {data.x.shape[0]} nodes, "
            f"raw features have {len(tx_id)} rows."
        )
    # The standard Elliptic PyG object has labels 0=unknown, 1=illicit, 2=licit.
    # Prefer its node ordering, while correcting the unknown convention.
    graph_y = data.y.detach().cpu().numpy().astype(np.int64)
    if np.all(np.isin(graph_y, [0, 1, 2])):
        labels = np.where(graph_y == 0, -1, np.where(graph_y == 1, 1, 0))
    return EllipticData(
        x=data.x.detach().cpu().float(),
        edge_index=data.edge_index.detach().cpu().long(),
        y=torch.from_numpy(labels),
        timestep=torch.from_numpy(timestep).long(),
        tx_id=tx_id,
        feature_names=feature_names,
    )


def make_temporal_splits(
    timestep: torch.Tensor,
    train_steps: Iterable[int] = range(1, 29),
    validation_steps: Iterable[int] = range(29, 35),
    test_steps: Iterable[int] = range(35, 50),
) -> TemporalSplits:
    """Return node masks for a chronological train/validation/test protocol."""
    train = torch.isin(timestep, torch.tensor(list(train_steps), dtype=timestep.dtype))
    validation = torch.isin(
        timestep, torch.tensor(list(validation_steps), dtype=timestep.dtype)
    )
    test = torch.isin(timestep, torch.tensor(list(test_steps), dtype=timestep.dtype))
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
