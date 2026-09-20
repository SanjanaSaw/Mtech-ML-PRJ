"""Consistent chronological RF/GCN/GAT evaluation.

This script preserves the original baseline architectures but applies the
same 1-28/29-34/35-49 protocol and validation threshold selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .data_utils import load_elliptic, make_temporal_splits
from .evaluation import binary_metrics, select_threshold
from .models import GAT, GCN


def run_rf(data, masks, trees=300):
    train = masks.train & (data.y >= 0)
    val = masks.validation & (data.y >= 0)
    test = masks.test & (data.y >= 0)
    model = RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(data.x[train].numpy(), data.y[train].numpy())
    pv = model.predict_proba(data.x[val].numpy())[:, 1]
    threshold = select_threshold(data.y[val].numpy(), pv)
    pt = model.predict_proba(data.x[test].numpy())[:, 1]
    row = binary_metrics(data.y[test].numpy(), pt, threshold)
    row["Model"] = "Random Forest"
    return row


def run_gnn(model_cls, data, masks, epochs=100, hidden_dim=128, lr=1e-3):
    model = model_cls(data.x.shape[1], hidden_dim, 2).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train, val, test = masks.train & (data.y >= 0), masks.validation & (data.y >= 0), masks.test & (data.y >= 0)
    counts = torch.bincount(data.y[train], minlength=2).float()
    loss = torch.nn.CrossEntropyLoss(weight=counts.sum() / counts.clamp_min(1))
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss(logits[train], data.y[train]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probability = torch.softmax(model(data.x, data.edge_index), -1)[:, 1].numpy()
    threshold = select_threshold(data.y[val].numpy(), probability[val.numpy()])
    row = binary_metrics(data.y[test].numpy(), probability[test.numpy()], threshold)
    row["Model"] = model_cls.__name__
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/elliptic")
    parser.add_argument("--output", default="results/model_comparison.csv")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    data = load_elliptic(args.data_root)
    masks = make_temporal_splits(data.timestep)
    rows = [run_rf(data, masks)]
    rows += [run_gnn(cls, data, masks, args.epochs) for cls in (GCN, GAT)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
