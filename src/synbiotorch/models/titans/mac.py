"""Causal Memory-as-Context block for the paper-deep TITANS realization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from synbiotorch.models.titans.memory import NeuralMemory
from synbiotorch.models.titans.state import NeuralMemoryState


def mac_attention_mask(persistent_tokens: int, segment_length: int, *, device: torch.device | None = None) -> Tensor:
    """Boolean ``[P; H; Z]`` mask where ``True`` blocks an attention edge."""

    layout_length = persistent_tokens + 2 * segment_length
    mask = torch.ones(layout_length, layout_length, dtype=torch.bool, device=device)
    mask[:persistent_tokens, :persistent_tokens] = False
    retrieval_start = persistent_tokens
    sequence_start = persistent_tokens + segment_length
    for position in range(segment_length):
        allowed = torch.cat(
            (
                torch.arange(persistent_tokens, device=device),
                torch.arange(retrieval_start, retrieval_start + position + 1, device=device),
                torch.arange(sequence_start, sequence_start + position + 1, device=device),
            )
        )
        mask[retrieval_start + position, allowed] = False
        mask[sequence_start + position, allowed] = False
    return mask


@dataclass(frozen=True)
class MACOutput:
    sequence: Tensor
    state: NeuralMemoryState
    retrieval: Tensor


class MACBlock(nn.Module):
    """Read one incoming fast state, attend causally, then optionally write it."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        persistent_tokens: int,
        segment_length: int,
        memory_expansion_factor: int,
        projection_kernel_size: int,
        alpha_initial: float,
        eta_initial: float,
        theta_initial: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.segment_length = segment_length
        self.persistent_token_count = persistent_tokens
        self.memory = NeuralMemory(
            d_model,
            expansion_factor=memory_expansion_factor,
            projection_kernel_size=projection_kernel_size,
            alpha_initial=alpha_initial,
            eta_initial=eta_initial,
            theta_initial=theta_initial,
        )
        self.persistent_tokens = nn.Parameter(torch.empty(persistent_tokens, d_model))
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.persistent_tokens, mean=0.0, std=0.02)

    def initial_state(self) -> NeuralMemoryState:
        return self.memory.initial_state()

    def integrate(self, retrieval: Tensor, segment: Tensor) -> Tensor:
        expected = (self.segment_length, self.d_model)
        if retrieval.shape != expected or segment.shape != expected:
            raise ValueError(f"retrieval and segment must have shape {expected}")
        persistent = self.persistent_tokens.to(segment)
        layout = torch.cat((persistent, retrieval.to(segment), segment), dim=0).unsqueeze(0)
        packed = F.linear(layout, self.attention.in_proj_weight, self.attention.in_proj_bias)
        query, key, value = packed.chunk(3, dim=-1)
        batch, length, width = query.shape
        heads = self.attention.num_heads
        head_width = width // heads

        def split(values: Tensor) -> Tensor:
            return values.view(batch, length, heads, head_width).transpose(1, 2)

        query, key, value = map(split, (query, key, value))
        blocked = mac_attention_mask(
            self.persistent_token_count,
            self.segment_length,
            device=layout.device,
        )
        additive = torch.zeros_like(blocked, dtype=layout.dtype).masked_fill(blocked, float("-inf"))
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_width**0.5)
        weights = torch.softmax(scores + additive.unsqueeze(0).unsqueeze(0), dim=-1)
        attended = torch.matmul(weights, value)
        merged = attended.transpose(1, 2).contiguous().view(batch, length, width)
        projected = F.linear(
            merged,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        sequence_start = self.persistent_token_count + self.segment_length
        return self.output_norm(segment + projected[0, sequence_start:])

    def forward(
        self,
        state: NeuralMemoryState,
        segment: Tensor,
        *,
        valid_mask: Tensor,
        update_memory: bool,
        create_graph: bool,
    ) -> MACOutput:
        retrieval, query_history = self.memory.read_with_history(state, segment)
        sequence = self.integrate(retrieval, segment)
        next_state = (
            self.memory.update(
                state,
                sequence,
                valid_mask=valid_mask,
                query_history=query_history,
                create_graph=create_graph,
            )
            if update_memory
            else self.memory.freeze(state, sequence, query_history=query_history)
        )
        return MACOutput(sequence, next_state, retrieval)
