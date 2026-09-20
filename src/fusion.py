"""Adaptive feature/graph/temporal fusion."""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveFusion(nn.Module):
    """Learn per-node softmax weights over feature, graph, and temporal views."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        feature_repr: torch.Tensor,
        graph_repr: torch.Tensor,
        temporal_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        views = torch.stack([feature_repr, graph_repr, temporal_repr], dim=1)
        weights = torch.softmax(self.gate(torch.cat([feature_repr, graph_repr, temporal_repr], dim=-1)), dim=-1)
        fused = (views * weights.unsqueeze(-1)).sum(dim=1)
        return self.output(fused), weights

