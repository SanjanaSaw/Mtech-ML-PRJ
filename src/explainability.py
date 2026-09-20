"""Optional, bounded explainability helpers for trained models."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch


def permutation_importance(
    model, x: torch.Tensor, edge_index: torch.Tensor, timestep: torch.Tensor,
    node_mask: torch.Tensor, y: torch.Tensor, repeats: int = 1,
) -> pd.DataFrame:
    """Estimate feature importance by shuffling one feature on a bounded mask."""
    from sklearn.metrics import average_precision_score
    model.eval()
    with torch.no_grad():
        baseline = torch.softmax(model(x, edge_index, timestep)["logits"], -1)[:, 1]
    baseline_score = average_precision_score(y[node_mask].cpu(), baseline[node_mask].cpu())
    rows = []
    for feature in range(x.shape[1]):
        drops = []
        for _ in range(repeats):
            permuted = x.clone()
            order = torch.randperm(int(node_mask.sum()), device=x.device)
            indices = torch.where(node_mask)[0]
            permuted[indices, feature] = permuted[indices[order], feature]
            with torch.no_grad():
                score = torch.softmax(model(permuted, edge_index, timestep)["logits"], -1)[:, 1]
            drops.append(baseline_score - average_precision_score(y[node_mask].cpu(), score[node_mask].cpu()))
        rows.append({"feature": feature, "importance": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def save_fusion_weights(path: str | Path, tx_id, timestep, weights: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = weights.detach().cpu().numpy()
    pd.DataFrame({
        "tx_id": tx_id, "timestep": timestep,
        "feature_weight": values[:, 0], "graph_weight": values[:, 1], "temporal_weight": values[:, 2],
    }).to_csv(path, index=False)
