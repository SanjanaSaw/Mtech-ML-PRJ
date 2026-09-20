"""TAF-GNN and configurable ablation variants."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from .fusion import AdaptiveFusion
from .temporal_model import TemporalEncoder


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kind: str = "gcn", heads: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.kind = kind.lower()
        conv = GATConv if self.kind == "gat" else GCNConv
        if self.kind == "gat":
            self.conv1 = conv(input_dim, hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = conv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        elif self.kind == "gcn":
            self.conv1 = conv(input_dim, hidden_dim)
            self.conv2 = conv(hidden_dim, hidden_dim)
        else:
            raise ValueError("kind must be 'gcn' or 'gat'")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.relu(self.conv2(x, edge_index))


class TAFGNN(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2,
        graph_type: str = "gcn", temporal_type: str = "gru", max_timestep: int = 49,
        use_graph: bool = True, use_temporal: bool = True, adaptive_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph, self.use_temporal, self.adaptive_fusion = use_graph, use_temporal, adaptive_fusion
        self.feature_encoder = FeatureEncoder(input_dim, hidden_dim, dropout)
        self.graph_encoder = GraphEncoder(input_dim, hidden_dim, graph_type, dropout=dropout) if use_graph else None
        self.temporal_encoder = TemporalEncoder(hidden_dim, temporal_type, dropout, max_timestep) if use_temporal else None
        self.fusion = AdaptiveFusion(hidden_dim, dropout) if adaptive_fusion else None
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, timestep: torch.Tensor) -> dict[str, torch.Tensor]:
        h_f = self.feature_encoder(x)
        h_g = self.graph_encoder(x, edge_index) if self.graph_encoder else h_f
        h_t = self.temporal_encoder(h_g, timestep) if self.temporal_encoder else h_g
        if self.fusion is not None:
            fused, weights = self.fusion(h_f, h_g, h_t)
        else:
            views = [h_f]
            if self.use_graph: views.append(h_g)
            if self.use_temporal: views.append(h_t)
            fused, weights = torch.stack(views).mean(dim=0), torch.empty(0, device=x.device)
        return {"logits": self.classifier(fused), "feature": h_f, "graph": h_g, "temporal": h_t, "fused": fused, "fusion_weights": weights}


# Keep the original public baseline classes available from models.py.
