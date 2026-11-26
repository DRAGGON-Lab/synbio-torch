"""Task heads attached on top of a backbone's pooled representation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from synbiotorch.config import TaskConfig


class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(pooled)).squeeze(-1)


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(pooled))


def build_head(task_config: "TaskConfig", pooled_dim: int, dropout: float) -> nn.Module:
    """Build the task head for a pooled representation of width ``pooled_dim``.

    ``pooled_dim`` comes from the pooler rather than from ``model.hidden_size``,
    since a pooler may widen or narrow the backbone's channel count.
    """
    if task_config.objective == "classification":
        assert task_config.num_classes is not None  # enforced by TaskConfig validation
        return ClassificationHead(pooled_dim, task_config.num_classes, dropout)
    return RegressionHead(pooled_dim, dropout)
