"""Reproducible chronological RF/GCN/GAT comparison and temporal analysis."""

from __future__ import annotations

import argparse
import copy
import json
import random
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from .data_utils import (
    filter_edges_by_time,
    load_elliptic,
    make_temporal_splits,
    validate_elliptic,
)
from .evaluation import binary_metrics, select_threshold, temporal_metrics
from .models import GAT, GCN


def parse_seeds(value):
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(value.split(",")):
        raise argparse.ArgumentTypeError("provide one or more comma-separated seeds")
    return seeds


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _period_probabilities(model, data, edge_index, mask, protocol, feature_count):
    probability = np.full(data.y.shape[0], np.nan, dtype=np.float64)
    if protocol == "transductive":
        with torch.no_grad():
            logits = model(data.x[:, :feature_count], edge_index)
            probability[mask.numpy()] = torch.softmax(logits, dim=-1)[mask, 1].cpu().numpy()
        return probability
    for step in torch.unique(data.timestep[mask], sorted=True).tolist():
        period_mask = mask & (data.timestep == step)
        period_edges = filter_edges_by_time(edge_index, data.timestep, int(step))
        with torch.no_grad():
            logits = model(data.x[:, :feature_count], period_edges)
            period_probability = torch.softmax(logits, dim=-1)[:, 1]
        probability[period_mask.numpy()] = period_probability[period_mask].cpu().numpy()
    return probability


def _fit_gnn(
    model_cls,
    data,
    masks,
    seed,
    epochs,
    hidden_dim,
    learning_rate,
    protocol,
    num_layers=3,
    feature_count=165,
    edge_mode="normal",
    patience=10,
):
    seed_everything(seed)
    model = model_cls(
        feature_count, hidden_dim, 2, num_layers=num_layers, dropout=0.2
    ).cpu()
    edges = data.edge_index.clone()
    if edge_mode == "none":
        edges = torch.empty((2, 0), dtype=torch.long)
    elif edge_mode == "shuffled":
        generator = torch.Generator().manual_seed(seed)
        edges[1] = edges[1, torch.randperm(edges.shape[1], generator=generator)]
    elif edge_mode != "normal":
        raise ValueError(f"Unknown edge mode: {edge_mode}.")
    train = masks.train & data.labeled_mask
    validation = masks.validation & data.labeled_mask
    test = masks.test & data.labeled_mask
    if torch.unique(data.y[train]).numel() != 2:
        raise ValueError("Training period must contain both labelled classes.")
    if torch.unique(data.y[validation]).numel() != 2:
        raise ValueError("Validation period must contain both labelled classes.")
    train_edges = (
        filter_edges_by_time(edges, data.timestep, int(data.timestep[train].max()))
        if protocol == "inductive"
        else edges
    )
    class_counts = torch.bincount(data.y[train], minlength=2).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_score, best_state, best_epoch, patience_left = -np.inf, None, 0, patience
    x = data.x[:, :feature_count]
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, train_edges)
        loss = criterion(logits[train], data.y[train])
        loss.backward()
        optimizer.step()
        model.eval()
        validation_probability = _period_probabilities(
            model, data, edges, validation, protocol, feature_count
        )
        selected = validation.numpy()
        score = binary_metrics(
            data.y[validation].numpy(), validation_probability[selected]
        )["PR-AUC"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected from validation PR-AUC.")
    model.load_state_dict(best_state)
    model.eval()
    val_probability = _period_probabilities(
        model, data, edges, validation, protocol, feature_count
    )
    test_probability = _period_probabilities(
        model, data, edges, test, protocol, feature_count
    )
    threshold = select_threshold(
        data.y[validation].numpy(), val_probability[validation.numpy()]
    )
    return model, val_probability, test_probability, threshold, best_epoch, edges


def _fit_rf(data, masks, seed, trees, feature_count):
    train = masks.train & data.labeled_mask
    validation = masks.validation & data.labeled_mask
    test = masks.test & data.labeled_mask
    if torch.unique(data.y[train]).numel() != 2:
        raise ValueError("Training period must contain both labelled classes.")
    if torch.unique(data.y[validation]).numel() != 2:
        raise ValueError("Validation period must contain both labelled classes.")
    model = RandomForestClassifier(
        n_estimators=trees, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    x = data.x[:, :feature_count].numpy()
    model.fit(x[train.numpy()], data.y[train].numpy())
    val_probability = np.full(data.y.shape[0], np.nan)
    test_probability = np.full(data.y.shape[0], np.nan)
    val_probability[validation.numpy()] = model.predict_proba(x[validation.numpy()])[:, 1]
    test_probability[test.numpy()] = model.predict_proba(x[test.numpy()])[:, 1]
    threshold = select_threshold(
        data.y[validation].numpy(), val_probability[validation.numpy()]
    )
    return model, val_probability, test_probability, threshold, 0, data.edge_index


def _fit_mlp(data, masks, seed, hidden_dim, feature_count):
    train = masks.train & data.labeled_mask
    validation = masks.validation & data.labeled_mask
    test = masks.test & data.labeled_mask
    if torch.unique(data.y[train]).numel() != 2:
        raise ValueError("Training period must contain both labelled classes.")
    if torch.unique(data.y[validation]).numel() != 2:
        raise ValueError("Validation period must contain both labelled classes.")
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        early_stopping=True,
        random_state=seed,
        max_iter=200,
    )
    x = data.x[:, :feature_count].numpy()
    model.fit(x[train.numpy()], data.y[train].numpy())
    val_probability = np.full(data.y.shape[0], np.nan)
    test_probability = np.full(data.y.shape[0], np.nan)
    val_probability[validation.numpy()] = model.predict_proba(x[validation.numpy()])[:, 1]
    test_probability[test.numpy()] = model.predict_proba(x[test.numpy()])[:, 1]
    threshold = select_threshold(
        data.y[validation].numpy(), val_probability[validation.numpy()]
    )
    return model, val_probability, test_probability, threshold, model.n_iter_, data.edge_index


def _precision_at_k(labels, probability, k=100):
    if len(labels) < k:
        return np.nan
    top = np.argpartition(probability, -k)[-k:]
    return float(np.asarray(labels)[top].mean())


def _recalibrated_threshold_rows(
    model_name, seed, data, masks, validation_probability, test_probability, fixed_threshold
):
    validation = masks.validation & data.labeled_mask
    validation_y = data.y[validation].numpy()
    validation_p = validation_probability[validation.numpy()]
    test = masks.test & data.labeled_mask
    test_steps = data.timestep[test].numpy()
    test_y = data.y[test].numpy()
    test_p = test_probability[test.numpy()]
    rows = []
    past_y, past_p = validation_y.copy(), validation_p.copy()
    for step in sorted(np.unique(test_steps)):
        current = test_steps == step
        for name, threshold in (
            ("validation_fixed", fixed_threshold),
            ("historical_recalibration", select_threshold(past_y, past_p)),
        ):
            metrics = binary_metrics(test_y[current], test_p[current], threshold)
            rows.append({
                "Model": model_name, "seed": seed, "timestep": int(step),
                "strategy": name, **metrics,
            })
        past_y = np.concatenate([past_y, test_y[current]])
        past_p = np.concatenate([past_p, test_p[current]])
    return rows


def _rolling_predictions(
    spec, data, step, seed, args, window, fixed_threshold, selected_epochs
):
    history = (data.timestep >= max(1, step - window)) & (data.timestep < step)
    history &= data.labeled_mask
    if torch.unique(data.y[history]).numel() != 2:
        raise ValueError(
            f"Rolling training window before timestep {step} lacks both labelled classes."
        )
    current = data.timestep == step
    if spec["type"] == "rf":
        model = RandomForestClassifier(
            n_estimators=args.trees, class_weight="balanced",
            random_state=seed + step, n_jobs=-1,
        )
        model.fit(data.x[history].numpy(), data.y[history].numpy())
        probability = model.predict_proba(data.x[current].numpy())[:, 1]
    else:
        model_cls = GCN if spec["type"] == "gcn" else GAT
        seed_everything(seed + step)
        model = model_cls(data.x.shape[1], args.hidden_dim, 2, num_layers=3).cpu()
        edges = (
            filter_edges_by_time(data.edge_index, data.timestep, step - 1)
            if args.protocol == "inductive" else data.edge_index
        )
        counts = torch.bincount(data.y[history], minlength=2).float()
        criterion = torch.nn.CrossEntropyLoss(weight=counts.sum() / counts.clamp_min(1))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for _ in range(max(1, selected_epochs)):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(data.x, edges)[history], data.y[history])
            loss.backward()
            optimizer.step()
        model.eval()
        period_edges = (
            filter_edges_by_time(data.edge_index, data.timestep, step)
            if args.protocol == "inductive" else data.edge_index
        )
        with torch.no_grad():
            logits = model(data.x, period_edges)
            probability = torch.softmax(logits, dim=-1)[current, 1].cpu().numpy()
    labels = data.y[current].numpy()
    metrics = binary_metrics(labels, probability, fixed_threshold)
    return {
        "Model": spec["name"], "seed": seed, "timestep": step,
        "training_nodes": int(history.sum()), "threshold": fixed_threshold,
        **{f"rolling_{key}": value for key, value in metrics.items()},
    }


def _temporal_distribution(data, output):
    steps = data.timestep.numpy()
    labels = data.y.numpy()
    rows = []
    for step in sorted(np.unique(steps)):
        nodes = steps == step
        labeled = nodes & (labels >= 0)
        rows.append({
            "timestep": int(step),
            "transaction_count": int(nodes.sum()),
            "labeled_count": int(labeled.sum()),
            "licit_count": int((labels[nodes] == 0).sum()),
            "illicit_count": int((labels[nodes] == 1).sum()),
            "illicit_proportion_labeled": (
                float((labels[nodes] == 1).sum() / labeled.sum())
                if labeled.any() else np.nan
            ),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "temporal_distribution.csv", index=False)
    return frame


def _feature_psi(data, masks, output, bins=10):
    features = data.x.numpy()
    train = features[masks.train.numpy()]
    if not len(train):
        raise ValueError("No labelled training rows available for feature drift.")
    quantiles = np.linspace(0, 1, bins + 1)
    rows = []
    for feature_index in range(features.shape[1]):
        edges = np.unique(np.quantile(train[:, feature_index], quantiles))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        reference = np.histogram(train[:, feature_index], bins=edges)[0].astype(float)
        reference = np.maximum(reference / reference.sum(), 1e-6)
        reference /= reference.sum()
        for step in sorted(np.unique(data.timestep.numpy())):
            values = features[(data.timestep == int(step)).numpy(), feature_index]
            if not len(values):
                continue
            observed = np.histogram(values, bins=edges)[0].astype(float)
            observed = np.maximum(observed / observed.sum(), 1e-6)
            observed /= observed.sum()
            psi = float(np.sum((observed - reference) * np.log(observed / reference)))
            rows.append({"timestep": int(step), "feature": feature_index + 1, "PSI": psi})
    result = pd.DataFrame(rows)
    result.to_csv(output / "feature_drift_psi.csv", index=False)
    return result


def _write_figures(temporal, distribution, drift, output):
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename in (
        ("ROC-AUC", "roc_auc_by_timestep.png"),
        ("Recall", "recall_by_timestep.png"),
        ("Precision", "precision_by_timestep.png"),
        ("PR-AUC", "pr_auc_by_timestep.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        summary = temporal.groupby(["Model", "timestep"])[metric].agg(
            ["mean", "std"]
        ).reset_index()
        for model, group in summary.groupby("Model"):
            ax.plot(group["timestep"], group["mean"], marker=".", label=model)
            spread = group["std"].fillna(0).to_numpy()
            ax.fill_between(
                group["timestep"],
                group["mean"].to_numpy() - spread,
                group["mean"].to_numpy() + spread,
                alpha=0.12,
            )
        ax.set(xlabel="Timestep", ylabel=metric, title=f"{metric} by timestep")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(distribution["timestep"], distribution["transaction_count"], marker=".")
    axes[0].set_ylabel("Transactions")
    axes[1].plot(
        distribution["timestep"],
        distribution["illicit_proportion_labeled"],
        marker=".",
    )
    axes[1].set(xlabel="Timestep", ylabel="Illicit proportion (labelled)")
    fig.tight_layout()
    fig.savefig(figure_dir / "transaction_distribution.png", dpi=160)
    plt.close(fig)
    mean_psi = drift.groupby("timestep")["PSI"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(mean_psi.index, mean_psi.values, marker=".")
    ax.set(xlabel="Timestep", ylabel="Mean feature PSI", title="Feature drift vs. training")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_drift_psi.png", dpi=160)
    plt.close(fig)


def _write_timestep_44_investigation(temporal, distribution, drift, output):
    nearby = temporal[temporal["timestep"].between(42, 46)].copy()
    class_counts = distribution[
        ["timestep", "transaction_count", "labeled_count", "illicit_proportion_labeled"]
    ].rename(columns={"transaction_count": "all_transaction_count"})
    mean_drift = drift.groupby("timestep", as_index=False)["PSI"].mean()
    nearby = nearby.merge(class_counts, on="timestep", how="left")
    nearby = nearby.merge(mean_drift, on="timestep", how="left")
    nearby.to_csv(output / "timestep_44_investigation.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/elliptic")
    parser.add_argument("--output", default="results/baselines/model_comparison.csv")
    parser.add_argument("--protocol", choices=["transductive", "inductive"], default="transductive")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seeds", type=parse_seeds, default=[42, 123, 2024, 2025, 2026])
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument(
        "--rolling-window", type=int, default=0,
        help="Enable period-by-period retraining on the previous N labelled timesteps.",
    )
    args = parser.parse_args(argv)
    if args.rolling_window < 0:
        parser.error("--rolling-window must be zero or greater.")
    data = load_elliptic(args.data_root)
    summary = validate_elliptic(data)
    masks = make_temporal_splits(data.timestep)
    output_file = Path(args.output)
    output = output_file.parent
    output.mkdir(parents=True, exist_ok=True)
    model_specs = [
        {"name": "Random Forest", "type": "rf", "features": 165},
        {"name": "GCN-3", "type": "gcn", "features": 165, "layers": 3},
        {"name": "GAT-3", "type": "gat", "features": 165, "layers": 3},
    ]
    if args.ablations:
        model_specs.extend([
            {"name": f"GCN-{depth}", "type": "gcn", "features": 165, "layers": depth}
            for depth in (1, 2)
        ])
        model_specs.extend([
            {"name": "GCN-3-no-edges", "type": "gcn", "features": 165, "layers": 3, "edges": "none"},
            {"name": "GCN-3-shuffled-edges", "type": "gcn", "features": 165, "layers": 3, "edges": "shuffled"},
            {"name": "GCN-3-first-94", "type": "gcn", "features": 94, "layers": 3},
            {"name": "Random Forest-first-94", "type": "rf", "features": 94},
            {"name": "MLP-feature-only", "type": "mlp", "features": 165},
        ])
    if summary["features"] < 165:
        raise ValueError("Elliptic feature ablations require the expected 165 input features.")
    per_seed_metrics, per_seed_temporal, per_seed_predictions, confusion_rows = [], [], [], []
    threshold_rows, rolling_rows = [], []
    for seed in args.seeds:
        seed_dir = output / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for spec in model_specs:
            started = time.perf_counter()
            model_cls = GCN if spec["type"] == "gcn" else GAT
            if spec["type"] == "rf":
                _, val_probability, test_probability, threshold, best_epoch, _ = _fit_rf(
                    data, masks, seed, args.trees, spec["features"]
                )
            elif spec["type"] == "mlp":
                _, val_probability, test_probability, threshold, best_epoch, _ = _fit_mlp(
                    data, masks, seed, args.hidden_dim, spec["features"]
                )
            else:
                _, val_probability, test_probability, threshold, best_epoch, _ = _fit_gnn(
                    model_cls, data, masks, seed, args.epochs, args.hidden_dim,
                    args.learning_rate, args.protocol, spec["layers"],
                    spec["features"], spec.get("edges", "normal"), args.patience,
                )
            test_mask = masks.test & data.labeled_mask
            selected = test_mask.numpy()
            labels = data.y[test_mask].numpy()
            probabilities = test_probability[selected]
            metrics = binary_metrics(labels, probabilities, threshold)
            metrics.update({
                "Model": spec["name"], "seed": seed, "protocol": args.protocol,
                "threshold": threshold, "selected_epoch": best_epoch,
                "Precision@100": _precision_at_k(labels, probabilities),
                "runtime_seconds": time.perf_counter() - started,
                "features": spec["features"], "edges": spec.get("edges", "normal"),
                "layers": spec.get("layers", 0),
                "graph_visibility": (
                    "complete graph (including future nodes and edges)"
                    if args.protocol == "transductive"
                    else "at each timestep, nodes and edges with both endpoints at or before that timestep"
                ),
            })
            per_seed_metrics.append(metrics)
            threshold_rows.extend(_recalibrated_threshold_rows(
                spec["name"], seed, data, masks, val_probability,
                test_probability, threshold,
            ))
            prediction = pd.DataFrame({
                "model": spec["name"], "seed": seed,
                "tx_id": data.tx_id[selected],
                "timestep": data.timestep.numpy()[selected],
                "y_true": labels, "probability": probabilities,
                "threshold": threshold, "prediction": probabilities >= threshold,
            })
            per_seed_predictions.append(prediction)
            temporal = temporal_metrics(
                labels, probabilities, data.timestep.numpy()[selected], threshold
            )
            temporal["Model"] = spec["name"]
            temporal["seed"] = seed
            temporal["protocol"] = args.protocol
            per_seed_temporal.append(temporal)
            if args.rolling_window and spec["name"] in {"Random Forest", "GCN-3", "GAT-3"}:
                for step in sorted(torch.unique(data.timestep[masks.test]).tolist()):
                    rolling_rows.append(_rolling_predictions(
                        spec, data, int(step), seed, args, args.rolling_window,
                        threshold, best_epoch,
                    ))
            tn, fp, fn, tp = confusion_matrix(labels, probabilities >= threshold, labels=[0, 1]).ravel()
            confusion_rows.append({
                "Model": spec["name"], "seed": seed, "threshold": threshold,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            })
            pd.DataFrame([confusion_rows[-1]]).to_csv(
                seed_dir / f"{spec['name'].lower().replace(' ', '_')}_confusion.csv",
                index=False,
            )
        pd.concat([p for p in per_seed_predictions if p["seed"].iloc[0] == seed]).to_csv(
            seed_dir / "predictions.csv", index=False
        )
    metrics_frame = pd.DataFrame(per_seed_metrics)
    temporal_frame = pd.concat(per_seed_temporal, ignore_index=True)
    predictions_frame = pd.concat(per_seed_predictions, ignore_index=True)
    metrics_frame.to_csv(output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrices.csv", index=False)
    temporal_frame.to_csv(output / "temporal_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output / "threshold_mitigations.csv", index=False)
    if rolling_rows:
        pd.DataFrame(rolling_rows).to_csv(output / "rolling_window_metrics.csv", index=False)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    temporal_distribution = _temporal_distribution(data, output)
    drift = _feature_psi(data, masks, output)
    _write_figures(temporal_frame, temporal_distribution, drift, output)
    _write_timestep_44_investigation(
        temporal_frame, temporal_distribution, drift, output
    )
    aggregate_metrics = [
        "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "MCC",
        "Balanced Accuracy", "False Positive Rate", "Precision@100",
    ]
    aggregate = metrics_frame.groupby(["Model", "protocol"], as_index=False).agg(
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in aggregate_metrics for stat in ("mean", "std")
        }
    )
    aggregate["seeds"] = ",".join(map(str, args.seeds))
    aggregate.to_csv(output_file, index=False)
    aggregate.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "dataset_summary.csv", index=False)
    (output / "config.json").write_text(
        json.dumps({
            "protocol": args.protocol,
            "train_timesteps": [1, 24],
            "validation_timesteps": [25, 34],
            "test_timesteps": [35, 49],
            "seeds": args.seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "random_forest_trees": args.trees,
            "ablations": args.ablations,
            "rolling_window": args.rolling_window,
            "models": model_specs,
            "checkpoint_selection_metric": "validation average precision (PR-AUC)",
            "threshold_selection_metric": "validation F1",
            "prediction_metrics": [
                "illicit precision", "illicit recall", "illicit F1",
                "ROC-AUC", "average precision", "Precision@100",
            ],
            "rolling_label_availability": (
                "Labels from completed prior periods are assumed available; "
                "the current prediction period is excluded."
            ),
            "strict_graph_visibility": (
                "Graph nodes and edges are restricted to endpoints at or before "
                "the prediction timestep."
            ),
            "first_94_features": "first 94 feature columns in raw file order; positional, no semantic group asserted",
            "dataset_summary": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
