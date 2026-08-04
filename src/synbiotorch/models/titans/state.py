"""Explicit functional state for the paper-deep TITANS MAC architecture."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

TensorMap = OrderedDict[str, Tensor]


@dataclass(frozen=True)
class NeuralMemoryState:
    """Fast memory weights, surprise momentum, and causal projection history."""

    fast_weights: Mapping[str, Tensor]
    surprise: Mapping[str, Tensor]
    query_history: Tensor
    write_history: Tensor

    def __post_init__(self) -> None:
        if tuple(self.fast_weights) != tuple(self.surprise):
            raise ValueError("fast weights and surprise must have identical keys")
        for name, value in self.fast_weights.items():
            if value.shape != self.surprise[name].shape:
                raise ValueError(f"surprise[{name!r}] must match its fast weight")
        if self.query_history.shape != self.write_history.shape or self.query_history.ndim != 2:
            raise ValueError("query and write histories must have equal two-dimensional shapes")

    def detach(self) -> "NeuralMemoryState":
        return NeuralMemoryState(
            fast_weights=OrderedDict(
                (name, value.detach().requires_grad_(True)) for name, value in self.fast_weights.items()
            ),
            surprise=OrderedDict((name, value.detach()) for name, value in self.surprise.items()),
            query_history=self.query_history.detach(),
            write_history=self.write_history.detach(),
        )

    def to(self, device: torch.device | str) -> "NeuralMemoryState":
        target = torch.device(device)
        return NeuralMemoryState(
            fast_weights=OrderedDict(
                (name, value.to(target).requires_grad_(True)) for name, value in self.fast_weights.items()
            ),
            surprise=OrderedDict((name, value.to(target)) for name, value in self.surprise.items()),
            query_history=self.query_history.to(target),
            write_history=self.write_history.to(target),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "fast_weights": OrderedDict(
                (name, value.detach().cpu().clone()) for name, value in self.fast_weights.items()
            ),
            "surprise": OrderedDict((name, value.detach().cpu().clone()) for name, value in self.surprise.items()),
            "query_history": self.query_history.detach().cpu().clone(),
            "write_history": self.write_history.detach().cpu().clone(),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "NeuralMemoryState":
        fast = payload.get("fast_weights")
        surprise = payload.get("surprise")
        query = payload.get("query_history")
        write = payload.get("write_history")
        if not isinstance(fast, Mapping) or not isinstance(surprise, Mapping):
            raise ValueError("invalid neural-memory tensor mappings")
        if not isinstance(query, Tensor) or not isinstance(write, Tensor):
            raise ValueError("invalid neural-memory projection histories")
        return cls(
            fast_weights=OrderedDict(
                (str(name), value.detach().clone().requires_grad_(True))
                for name, value in fast.items()
                if isinstance(value, Tensor)
            ),
            surprise=OrderedDict(
                (str(name), value.detach().clone()) for name, value in surprise.items() if isinstance(value, Tensor)
            ),
            query_history=query.detach().clone(),
            write_history=write.detach().clone(),
        )


@dataclass(frozen=True)
class TitansState:
    """The runtime fast state for every MAC block of one logical sequence."""

    blocks: tuple[NeuralMemoryState, ...]

    def detach(self) -> "TitansState":
        return TitansState(tuple(block.detach() for block in self.blocks))

    def to(self, device: torch.device | str) -> "TitansState":
        return TitansState(tuple(block.to(device) for block in self.blocks))

    def state_dict(self) -> dict[str, object]:
        return {"format_version": 1, "blocks": [block.state_dict() for block in self.blocks]}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "TitansState":
        if payload.get("format_version") != 1:
            raise ValueError("unsupported TITANS state format")
        blocks = payload.get("blocks")
        if not isinstance(blocks, list) or not all(isinstance(block, Mapping) for block in blocks):
            raise ValueError("invalid TITANS block-state payload")
        return cls(tuple(NeuralMemoryState.from_state_dict(block) for block in blocks))
