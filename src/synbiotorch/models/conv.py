"""The architecture plug point, and the convolutional family.

``SequenceBackbone`` is the base class every non-HuggingFace sequence
architecture implements. It is a class rather than a ``Protocol`` on purpose: a
HuggingFace module structurally satisfies ``forward(input_ids, attention_mask)``
too, so a structural check could not tell the two apart, and ``SequenceModel``
needs that distinction to know how to read the output.

``ConvBackbone`` is a 1D convolutional motif scanner. Convolution plus pooling
is position-invariant by construction, which is the right inductive bias for
regulatory DNA: promoter elements are short motifs appearing at variable
offsets, often with a tolerance-bounded gap between them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from synbiotorch.config import ConvArch

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


class SequenceBackbone(nn.Module, ABC):
    """Maps a padded token batch to per-position features.

    Implementations return features **channels-last** as ``[B, L', C]`` so a
    single set of poolers serves every family, together with the attention mask
    for ``L'``. Returning the mask is what makes the contract general: a
    convolution or pooling stack changes the sequence length, and anything
    downstream that masks needs the new one, not the input's.
    """

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Channel count of the returned features."""

    @abstractmethod
    def out_length(self, in_length: int) -> int:
        """Output length for a given input length, as pure arithmetic.

        Lets the head be sized before any tensor exists, and is independently
        testable.
        """


class ConvBackbone(SequenceBackbone):
    """Embedding, then a stack of valid 1D convolutions, then one max-pool.

    The embedding is learned rather than a fixed one-hot. At the default width
    of 4 it can represent one-hot ACGT exactly, so it is a strict
    generalization, and it avoids hard-coding a tokenizer's id layout into the
    model.
    """

    def __init__(self, arch: ConvArch, *, vocab_size: int, pad_token_id: int, dropout: float) -> None:
        super().__init__()
        self.arch = arch
        self.embedding = nn.Embedding(vocab_size, arch.embed_dim, padding_idx=pad_token_id)
        activation = _ACTIVATIONS[arch.activation]
        layers: list[nn.Module] = []
        in_channels = arch.embed_dim
        for channels, width in zip(arch.channels, arch.kernel_sizes):
            layers += [nn.Conv1d(in_channels, channels, width), activation(), nn.Dropout(dropout)]
            in_channels = channels
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool1d(arch.pool_width)
        self._out_channels = in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def out_length(self, in_length: int) -> int:
        length = in_length
        for width in self.arch.kernel_sizes:
            length = length - width + 1
            if length < 1:
                raise ValueError(
                    f"input of {in_length} positions is too short for kernels {list(self.arch.kernel_sizes)}"
                )
        return length // self.arch.pool_width

    def _downsample_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Carry the mask through the same shape changes as the features.

        The two steps have genuinely different semantics. A valid convolution
        output needs its whole receptive field valid, so the mask *erodes*; a
        max-pool bin is valid if any position in it is, so the mask *dilates*
        into the bin.
        """
        mask = attention_mask.unsqueeze(1).float()
        for width in self.arch.kernel_sizes:
            mask = (F.avg_pool1d(mask, width, stride=1) == 1.0).float()
        mask = (F.max_pool1d(mask, self.arch.pool_width) > 0).float()
        return mask.squeeze(1).to(attention_mask.dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Zero the padding before convolving so it cannot leak into a valid
        # receptive field, then work channels-first as Conv1d expects.
        embedded = self.embedding(input_ids) * attention_mask.unsqueeze(-1).type_as(self.embedding.weight)
        hidden = self.pool(self.convs(embedded.transpose(1, 2)))
        return hidden.transpose(1, 2), self._downsample_mask(attention_mask)


def build_conv_backbone(
    arch: ConvArch, *, vocab_size: int, pad_token_id: int, dropout: float
) -> tuple[ConvBackbone, int]:
    """Return ``(backbone, out_channels)`` for the conv family."""
    backbone = ConvBackbone(arch, vocab_size=vocab_size, pad_token_id=pad_token_id, dropout=dropout)
    return backbone, backbone.out_channels
