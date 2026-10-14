import numpy as np
from types import SimpleNamespace
import torch

from src.data_utils import EllipticData, make_temporal_splits
from src.run_experiments import _predict_mask, _visible_graph, train_variant
from src.baseline_experiments import _period_probabilities, _recalibrated_threshold_rows
from src.evaluation import select_threshold
from src.models import GCN
from src.taf_gnn import TAFGNN


def _small_data():
    return EllipticData(
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        y=torch.tensor([0, 1, 0, 1, 1, 1, 0]),
        timestep=torch.tensor([1, 1, 25, 26, 35, 35, 36]),
        tx_id=np.array(["a", "b", "c", "d", "e", "f", "g"]),
        feature_names=["feature_1"],
    )


def test_inductive_subgraph_contains_only_visible_nodes_and_edges():
    data = _small_data()
    x, edges, timestep, visible_nodes, global_to_local = _visible_graph(
        data, data.edge_index, cutoff=25, device=torch.device("cpu")
    )

    assert visible_nodes.tolist() == [0, 1, 2]
    assert x[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert timestep.tolist() == [1, 1, 25]
    assert edges.tolist() == [[0, 1], [1, 2]]
    assert global_to_local.tolist() == [0, 1, 2, -1, -1, -1, -1]


def test_future_features_and_edges_cannot_change_earlier_inductive_predictions():
    torch.manual_seed(7)
    data = _small_data()
    model = TAFGNN(
        input_dim=1, hidden_dim=4, dropout=0.0, use_graph=True,
        use_temporal=True, adaptive_fusion=True,
    )
    current = (data.timestep == 35) & data.labeled_mask
    original, _ = _predict_mask(
        model, data, data.edge_index, current, "inductive", torch.device("cpu")
    )

    changed_x = data.x.clone()
    changed_x[data.timestep > 35] = 1000
    future_edge = torch.tensor([[6], [4]])
    changed_data = EllipticData(
        x=changed_x,
        edge_index=torch.cat([data.edge_index, future_edge], dim=1),
        y=data.y,
        timestep=data.timestep,
        tx_id=data.tx_id,
        feature_names=data.feature_names,
    )
    changed, _ = _predict_mask(
        model, changed_data, changed_data.edge_index, current,
        "inductive", torch.device("cpu"),
    )

    np.testing.assert_allclose(original[current.numpy()], changed[current.numpy()])


def test_baseline_gnn_compacts_future_nodes_during_inductive_inference():
    torch.manual_seed(11)
    data = _small_data()
    model = GCN(in_channels=1, hidden_channels=4, out_channels=2, num_layers=2, dropout=0.0)
    current = data.timestep == 35
    original = _period_probabilities(
        model, data, data.edge_index, current, "inductive", feature_count=1
    )

    changed_x = data.x.clone()
    changed_x[data.timestep > 35] = 1000
    future_edge = torch.tensor([[6], [4]])
    changed_data = EllipticData(
        x=changed_x,
        edge_index=torch.cat([data.edge_index, future_edge], dim=1),
        y=data.y,
        timestep=data.timestep,
        tx_id=data.tx_id,
        feature_names=data.feature_names,
    )
    changed = _period_probabilities(
        model, changed_data, changed_data.edge_index, current,
        "inductive", feature_count=1,
    )

    np.testing.assert_allclose(original[current.numpy()], changed[current.numpy()])
    assert np.isnan(original[~current.numpy()]).all()


def test_taf_training_uses_validation_predictions_for_threshold_selection():
    data = _small_data()
    masks = make_temporal_splits(data.timestep)
    args = SimpleNamespace(
        device="cpu", hidden_dim=4, dropout=0.0, graph_type="gcn",
        temporal_type="gru", protocol="inductive", focal_gamma=2.0,
        learning_rate=0.01, weight_decay=0.0, patience=1, epochs=1,
    )
    variant = {
        "graph": False, "temporal": False, "fusion": False,
        "loss": "cross_entropy", "name": "test-feature-only",
    }

    _, _, probability, _, threshold, test_mask = train_variant(
        args, data, masks, variant, data.edge_index
    )

    assert np.isfinite(threshold)
    assert np.isfinite(probability[test_mask]).all()
    assert np.isnan(probability[~test_mask]).all()


def test_test_period_recalibration_never_uses_current_period_labels():
    data = _small_data()
    masks = make_temporal_splits(
        data.timestep, train_steps=[1], validation_steps=[25, 26], test_steps=[35, 36]
    )
    validation_probability = np.array([np.nan, np.nan, 0.1, 0.3, np.nan, np.nan, np.nan])
    test_probability = np.array([np.nan, np.nan, np.nan, np.nan, 0.8, 0.2, 0.7])
    fixed_threshold = select_threshold(
        data.y[masks.validation].numpy(),
        validation_probability[masks.validation.numpy()],
    )

    rows = _recalibrated_threshold_rows(
        "RF", 42, data, masks, validation_probability, test_probability, fixed_threshold
    )
    first_period = [row for row in rows if row["timestep"] == 35]

    assert first_period[0]["threshold"] == fixed_threshold
    assert first_period[1]["threshold"] == fixed_threshold

    changed_y = data.y.clone()
    changed_y[data.timestep == 35] = 0
    changed_y[data.timestep == 36] = 1
    changed_data = EllipticData(
        x=data.x, edge_index=data.edge_index, y=changed_y,
        timestep=data.timestep, tx_id=data.tx_id, feature_names=data.feature_names,
    )
    changed_rows = _recalibrated_threshold_rows(
        "RF", 42, changed_data, masks,
        validation_probability, test_probability, fixed_threshold,
    )
    thresholds = [(row["timestep"], row["strategy"], row["threshold"]) for row in rows]
    changed_thresholds = [
        (row["timestep"], row["strategy"], row["threshold"]) for row in changed_rows
    ]
    assert changed_thresholds == thresholds
