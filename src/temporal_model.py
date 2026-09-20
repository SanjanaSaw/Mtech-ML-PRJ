"""Temporal encoders used by TAF-GNN."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """Encode one representation per timestep and broadcast it to its nodes."""

    def __init__(
        self,
        hidden_dim: int,
        encoder: str = "gru",
        dropout: float = 0.2,
        max_timestep: int = 49,
    ) -> None:
        super().__init__()
        if encoder.lower() not in {"gru", "lstm"}:
            raise ValueError("encoder must be 'gru' or 'lstm'")
        self.encoder_name = encoder.lower()
        self.time_embedding = nn.Embedding(max_timestep + 1, hidden_dim)
        recurrent = nn.GRU if self.encoder_name == "gru" else nn.LSTM
        self.recurrent = recurrent(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, representation: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        unique_steps = torch.unique(timestep, sorted=True)
        pooled = []
        for step in unique_steps:
            mask = timestep == step
            pooled.append(representation[mask].mean(dim=0))
        sequence = torch.stack(pooled).unsqueeze(0)
        sequence = sequence + self.time_embedding(unique_steps).unsqueeze(0)
        output, _ = self.recurrent(sequence)
        step_output = output.squeeze(0)
        positions = torch.searchsorted(unique_steps, timestep)
        return self.dropout(step_output[positions])

