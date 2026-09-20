"""Reproducible training/evaluation entry point for the research experiments.

Example:
    python -m src.run_experiments --data-root data/elliptic --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from .data_utils import filter_edges_by_time, load_elliptic, make_temporal_splits
from .evaluation import binary_metrics, save_predictions, select_threshold, temporal_metrics
from .explainability import save_fusion_weights
from .losses import build_loss
from .taf_gnn import TAFGNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_variant(args, data, masks, variant, edge_index):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TAFGNN(
        data.x.shape[1], args.hidden_dim, args.dropout, graph_type=args.graph_type,
        temporal_type=args.temporal_type, use_graph=variant["graph"],
        use_temporal=variant["temporal"], adaptive_fusion=variant["fusion"],
    ).to(device)
    x, edges, timestep, y = data.x.to(device), edge_index.to(device), data.timestep.to(device), data.y.to(device)
    train_mask, val_mask, test_mask = [m.to(device) for m in (masks.train, masks.validation, masks.test)]
    known_train = train_mask & (y >= 0)
    known_val = val_mask & (y >= 0)
    known_test = test_mask & (y >= 0)
    counts = torch.bincount(y[known_train], minlength=2).float()
    weights = (counts.sum() / counts.clamp_min(1)).to(device)
    loss = build_loss(variant["loss"], weights, args.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_score, best_state, patience_left = -np.inf, None, args.patience
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x, edges, timestep)
        loss_value = loss(output["logits"][known_train], y[known_train])
        loss_value.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(x, edges, timestep)["logits"], -1)[:, 1]
        val_score = float(binary_metrics(y[known_val].cpu().numpy(), probability[known_val].cpu().numpy())["PR-AUC"])
        if val_score > best_score:
            best_score, best_state, patience_left = val_score, copy.deepcopy(model.state_dict()), args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected; validation labels may be empty.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(x, edges, timestep)
        probability = torch.softmax(output["logits"], -1)[:, 1].cpu().numpy()
    threshold = select_threshold(y[known_val].cpu().numpy(), probability[known_val.cpu().numpy()])
    metrics = binary_metrics(y[known_test].cpu().numpy(), probability[known_test.cpu().numpy()], threshold)
    metrics.update({"Model": variant["name"], "training_seconds": time.perf_counter() - start,
                    "parameters": sum(p.numel() for p in model.parameters())})
    return model, output, probability, metrics, threshold, known_test.cpu().numpy()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/elliptic")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--graph-type", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--temporal-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--device", default="")
    parser.add_argument("--protocol", choices=["transductive", "inductive"], default="transductive")
    parser.add_argument("--seed", type=int, default=42)
    parsed = parser.parse_args(args)
    seed_everything(parsed.seed)
    data = load_elliptic(parsed.data_root)
    masks = make_temporal_splits(data.timestep)
    out = Path(parsed.output_root)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    # The transductive protocol intentionally uses the complete graph.
    edge_index = data.edge_index
    if parsed.protocol == "inductive":
        edge_index = filter_edges_by_time(edge_index, data.timestep, max_timestep=34)
    variants = [
        {"name": "M1 Feature-only", "graph": False, "temporal": False, "fusion": False, "loss": "cross_entropy"},
        {"name": "M2 Feature+Graph", "graph": True, "temporal": False, "fusion": False, "loss": "weighted_cross_entropy"},
        {"name": "M3 Feature+Graph+Temporal", "graph": True, "temporal": True, "fusion": False, "loss": "weighted_cross_entropy"},
        {"name": "M4 TAF-GNN", "graph": True, "temporal": True, "fusion": True, "loss": "weighted_cross_entropy"},
        {"name": "M5 TAF-GNN+Focal", "graph": True, "temporal": True, "fusion": True, "loss": "focal"},
    ]
    rows, final = [], None
    for variant in variants:
        model, output, probability, metrics, threshold, test_mask = train_variant(
            parsed, data, masks, variant, edge_index
        )
        rows.append({k: metrics[k] for k in ["Model", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "MCC", "Balanced Accuracy", "False Positive Rate"]})
        if variant["name"] == "M4 TAF-GNN":
            final = (model, output, probability, metrics, threshold, test_mask)
    if final is None:
        raise RuntimeError("TAF-GNN variant did not run.")
    model, output, probability, metrics, threshold, test_mask = final
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ablation_results.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)
    labeled_test = masks.test.numpy() & (data.y.numpy() >= 0)
    save_predictions(out / "taf_gnn_predictions.csv", data.tx_id[labeled_test], data.timestep.numpy()[labeled_test],
                     data.y.numpy()[labeled_test], probability[labeled_test], threshold)
    temporal_df = temporal_metrics(
        data.y.numpy()[labeled_test], probability[labeled_test],
        data.timestep.numpy()[labeled_test], threshold
    )
    temporal_df.to_csv(out / "temporal_results.csv", index=False)
    save_fusion_weights(out / "fusion_weights.csv", data.tx_id, data.timestep.numpy(), output["fusion_weights"])
    y_test = data.y.numpy()[labeled_test]
    p_test = probability[labeled_test]
    pred_test = (p_test >= threshold).astype(int)
    categories = np.where((pred_test == 1) & (y_test == 1), "true_positive",
                 np.where((pred_test == 1) & (y_test == 0), "false_positive",
                 np.where(y_test == 1, "false_negative", "true_negative")))
    pd.DataFrame({
        "tx_id": data.tx_id[labeled_test], "timestep": data.timestep.numpy()[labeled_test],
        "true_label": y_test, "predicted_probability": p_test,
        "predicted_label": pred_test, "error_category": categories,
    }).to_csv(out / "error_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    RocCurveDisplay.from_predictions(y_test, p_test, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=axes[1])
    axes[0].set_title("TAF-GNN ROC curve")
    axes[1].set_title("TAF-GNN precision-recall curve")
    fig.tight_layout()
    fig.savefig(out / "figures/taf_gnn_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(temporal_df["timestep"], temporal_df["F1"], marker="o", label="F1")
    ax.plot(temporal_df["timestep"], temporal_df["PR-AUC"], marker="s", label="PR-AUC")
    ax.set_title("TAF-GNN Performance Across Future Timesteps")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures/temporal_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    weights = output["fusion_weights"].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Features", "Graph", "Temporal"], weights.mean(axis=0))
    ax.set_title("Mean TAF-GNN Fusion Weights")
    ax.set_ylabel("Mean softmax gate")
    fig.tight_layout()
    fig.savefig(out / "figures/fusion_weights.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    Path("models").mkdir(exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "threshold": threshold, "config": vars(parsed)}, "models/taf_gnn_best.pt")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
