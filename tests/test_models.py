import pytest
import torch

from src.models import GAT, GCN


@pytest.mark.parametrize("model_class", [GCN, GAT])
@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_gnn_depth_variants_return_binary_node_logits(model_class, num_layers):
    model = model_class(
        in_channels=4,
        hidden_channels=8,
        out_channels=2,
        num_layers=num_layers,
        dropout=0.0,
    )
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

    assert model(x, edge_index).shape == (5, 2)


@pytest.mark.parametrize("model_class", [GCN, GAT])
def test_gnn_rejects_zero_layers(model_class):
    with pytest.raises(ValueError, match="at least 1"):
        model_class(4, 8, 2, num_layers=0)
