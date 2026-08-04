"""Token-level causal language model built from paper-deep TITANS MAC blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, cast

import torch
from torch import Tensor, nn

from synbiotorch.config import TitansConfig
from synbiotorch.models.titans.mac import MACBlock
from synbiotorch.models.titans.state import TitansState

MemoryMode = Literal["adaptive", "frozen", "none"]


@dataclass(frozen=True)
class TitansCausalOutput:
    logits: Tensor
    states: tuple[TitansState, ...]


class TitansCausalLM(nn.Module):
    """Pure state-in/state-out MAC language model; stream identity stays outside."""

    def __init__(
        self,
        config: TitansConfig,
        *,
        vocab_size: int,
        pad_token_id: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(config.segment_length, d_model)
        self.blocks = nn.ModuleList(
            MACBlock(
                d_model,
                num_heads=config.num_attention_heads,
                persistent_tokens=config.persistent_tokens,
                segment_length=config.segment_length,
                memory_expansion_factor=config.memory_expansion_factor,
                projection_kernel_size=config.projection_kernel_size,
                alpha_initial=config.alpha_initial,
                eta_initial=config.eta_initial,
                theta_initial=config.theta_initial,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embeddings.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def initial_state(self) -> TitansState:
        return TitansState(tuple(cast(MACBlock, block).initial_state() for block in self.blocks))

    def forward_segment(
        self,
        input_ids: Tensor,
        states: Sequence[TitansState] | None = None,
        *,
        valid_mask: Tensor | None = None,
        memory_mode: MemoryMode = "adaptive",
    ) -> TitansCausalOutput:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, segment_length = input_ids.shape
        if segment_length != self.config.segment_length:
            raise ValueError(f"TITANS segments must have width {self.config.segment_length}, got {segment_length}")
        if states is None:
            states = tuple(self.initial_state() for _ in range(batch_size))
        if len(states) != batch_size:
            raise ValueError("one TITANS state is required per batch row")
        if valid_mask is None:
            valid_mask = input_ids.ne(self.pad_token_id)
        if valid_mask.shape != input_ids.shape:
            raise ValueError("valid_mask must match input_ids")
        if memory_mode not in {"adaptive", "frozen", "none"}:
            raise ValueError(f"unknown TITANS memory mode: {memory_mode}")

        positions = torch.arange(segment_length, device=input_ids.device)
        create_graph = self.training and torch.is_grad_enabled()
        logits: list[Tensor] = []
        next_rows: list[TitansState] = []
        for row, state in enumerate(states):
            if len(state.blocks) != len(self.blocks):
                raise ValueError("TITANS state block count does not match the model")
            sequence = self.token_embeddings(input_ids[row]) * math.sqrt(self.d_model) + self.position_embeddings(
                positions
            )
            next_blocks = []
            for module, block_state in zip(self.blocks, state.blocks):
                block = cast(MACBlock, module)
                if memory_mode == "none":
                    sequence = block.integrate(torch.zeros_like(sequence), sequence)
                    next_blocks.append(block_state)
                    continue
                output = block(
                    block_state,
                    sequence,
                    valid_mask=valid_mask[row].bool(),
                    update_memory=memory_mode == "adaptive",
                    create_graph=create_graph,
                )
                sequence = output.sequence
                next_blocks.append(output.state)
            logits.append(self.lm_head(self.final_norm(sequence)))
            next_rows.append(TitansState(tuple(next_blocks)))
        return TitansCausalOutput(torch.stack(logits), tuple(next_rows))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Fresh-state convenience path for architecture probes, not long streams."""

        return self.forward_segment(input_ids, valid_mask=attention_mask).logits

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
