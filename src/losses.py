"""Imbalance-aware losses."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else torch.tensor([]))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha if self.alpha.numel() else None, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt).pow(self.gamma)) * ce).mean()


def build_loss(
    name: str = "weighted_cross_entropy",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss()
    if normalized in {"weighted_cross_entropy", "weighted_ce"}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if normalized in {"focal", "focal_loss"}:
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    raise ValueError(f"Unknown loss: {name}")

