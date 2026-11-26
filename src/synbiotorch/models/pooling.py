"""Poolers — reduce a backbone's per-position features to one vector per record.

Separating pooling from the backbone is what lets one set of task heads serve
every architecture family: a transformer wants masked mean pooling over tokens,
while a convolutional motif scanner wants to keep the positional layout its
filters produced. Each pooler reports the width it yields so the head can be
sized before any data exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from synbiotorch.exceptions import ConfigError


class Pooler(nn.Module, ABC):
    """Reduces ``[B, L, C]`` features plus a mask to one ``[B, D]`` vector."""

    @abstractmethod
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def output_dim(self, channels: int) -> int:
        """Width this pooler yields, so the head can be sized before any data exists."""


def _masked_fill_value(hidden: torch.Tensor) -> float:
    """Lowest finite value of the tensor's dtype.

    ``-inf`` would propagate to NaN through the head under fp16 autocast, so
    masked positions are filled with the dtype minimum instead.
    """
    return torch.finfo(hidden.dtype).min


class MeanPooler(Pooler):
    """Attention-masked mean over positions.

    The historical default, and the only pooling the library had. Works for
    backbones (like DNABERT-2) that expose no dedicated pooler output.
    """

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=torch.finfo(hidden.dtype).tiny)
        return summed / counts

    def output_dim(self, channels: int) -> int:
        return channels


class MaxPooler(Pooler):
    """Attention-masked max over positions — length-agnostic, drops position."""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        pooled = hidden.masked_fill(~mask, _masked_fill_value(hidden)).max(dim=1).values
        # A row with no valid position would otherwise emit the dtype minimum
        # straight into the head.
        any_valid = mask.any(dim=1).type_as(pooled)
        return pooled * any_valid

    def output_dim(self, channels: int) -> int:
        return channels


class ClsPooler(Pooler):
    """First position, for backbones trained with a leading ``<cls>`` token."""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0]

    def output_dim(self, channels: int) -> int:
        return channels


class FlattenPooler(Pooler):
    """Concatenate every position, preserving the positional layout.

    What a convolutional motif scanner needs: the spacing between motifs is
    signal, and averaging or maxing over positions discards it. The output
    width is fixed at ``channels * length``, so the batch must be padded to a
    constant length rather than to the longest member (see ``Collator.pad_to``)
    — otherwise a record's features would depend on its batch-mates.
    """

    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] != self.length:
            raise ValueError(
                f"FlattenPooler expects a fixed length of {self.length}, got {hidden.shape[1]}; "
                "the batch must be padded to the encoder's max_length"
            )
        masked = hidden * attention_mask.unsqueeze(-1).type_as(hidden)
        return masked.flatten(start_dim=1)

    def output_dim(self, channels: int) -> int:
        return channels * self.length


def build_pooler(kind: str, *, length: int | None = None) -> Pooler:
    """Construct the pooler named by ``model.arch.pooling``."""
    if kind == "mean":
        return MeanPooler()
    if kind == "max":
        return MaxPooler()
    if kind == "cls":
        return ClsPooler()
    if kind == "flatten":
        if length is None:
            raise ConfigError("pooling: flatten needs a known sequence length; set tokenizer.max_length")
        return FlattenPooler(length)
    raise ConfigError(f"unknown pooling '{kind}'")
