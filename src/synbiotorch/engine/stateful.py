"""Batch adapter for ordered stateful causal-language-model windows."""

from __future__ import annotations

from typing import Mapping, Sequence, cast

import torch
from torch import Tensor

from synbiotorch.models.titans import TitansCausalLM, TitansState


class StatefulCausalBatchAdapter:
    """Carry stream-keyed TITANS state across gradient-horizon windows."""

    FORMAT_VERSION = 1
    label_key = "labels"

    def __init__(self, *, memory_mode: str = "adaptive") -> None:
        if memory_mode not in {"adaptive", "frozen", "none"}:
            raise ValueError(f"unknown TITANS memory mode: {memory_mode}")
        self.memory_mode = memory_mode
        self.stream_states: dict[str, TitansState] = {}

    def reset(self) -> None:
        self.stream_states = {}

    def to_device(
        self, batch: dict[str, Tensor | tuple[str, ...]], device: torch.device
    ) -> dict[str, Tensor | tuple[str, ...]]:
        return {name: value.to(device) if isinstance(value, Tensor) else value for name, value in batch.items()}

    def labels(self, batch: dict[str, Tensor | tuple[str, ...]]) -> Tensor:
        labels = batch[self.label_key]
        if not isinstance(labels, Tensor):
            raise TypeError("stateful causal labels must be a tensor")
        return labels

    def forward(
        self,
        model: torch.nn.Module,
        batch: dict[str, Tensor | tuple[str, ...]],
    ) -> Tensor:
        if not isinstance(model, TitansCausalLM):
            raise TypeError("StatefulCausalBatchAdapter requires TitansCausalLM")
        stream_ids = batch["stream_ids"]
        input_ids = batch["input_ids"]
        valid_mask = batch["valid_mask"]
        segment_counts = batch["segment_counts"]
        starts = batch["start_of_stream"]
        ends = batch["end_of_stream"]
        if not isinstance(stream_ids, Sequence) or isinstance(stream_ids, (str, bytes)):
            raise TypeError("stream_ids must be a sequence of strings")
        if not all(isinstance(value, str) for value in stream_ids):
            raise TypeError("stream_ids must be a sequence of strings")
        tensors = (input_ids, valid_mask, segment_counts, starts, ends)
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("stateful causal batch fields must be tensors")
        input_ids = cast(Tensor, input_ids)
        valid_mask = cast(Tensor, valid_mask)
        segment_counts = cast(Tensor, segment_counts)
        starts = cast(Tensor, starts)
        ends = cast(Tensor, ends)
        batch_size, horizon, segment_length = input_ids.shape
        if batch_size != len(stream_ids):
            raise ValueError("stream identity count does not match batch rows")

        rows: list[Tensor] = []
        for row, stream_id in enumerate(stream_ids):
            if bool(starts[row].item()):
                if stream_id in self.stream_states:
                    raise RuntimeError(f"stream {stream_id!r} attempted to inherit existing state")
                state = model.initial_state()
            else:
                if stream_id not in self.stream_states:
                    raise RuntimeError(f"continuing stream {stream_id!r} has no state")
                state = self.stream_states[stream_id].to(input_ids.device)
            count = int(segment_counts[row].item())
            if not 1 <= count <= horizon:
                raise ValueError("segment count must be within the gradient horizon")
            row_logits: list[Tensor] = []
            for segment_index in range(count):
                output = model.forward_segment(
                    input_ids[row, segment_index].unsqueeze(0),
                    (state,),
                    valid_mask=valid_mask[row, segment_index].unsqueeze(0),
                    memory_mode=self.memory_mode,  # type: ignore[arg-type]
                )
                row_logits.append(output.logits[0])
                state = output.states[0]
            if count < horizon:
                padding = model.token_embeddings.weight.new_zeros((horizon - count, segment_length, model.vocab_size))
                row_logits.extend(padding[index] for index in range(horizon - count))
            rows.append(torch.stack(row_logits))
            if bool(ends[row].item()):
                self.stream_states.pop(stream_id, None)
            else:
                self.stream_states[stream_id] = state.detach()
        return torch.stack(rows)

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "memory_mode": self.memory_mode,
            "stream_states": {stream_id: state.state_dict() for stream_id, state in sorted(self.stream_states.items())},
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported stateful adapter checkpoint format")
        if payload.get("memory_mode") != self.memory_mode:
            raise ValueError("TITANS memory mode changed across resume")
        raw_states = payload.get("stream_states")
        if not isinstance(raw_states, Mapping):
            raise ValueError("stateful adapter checkpoint lacks stream states")
        restored: dict[str, TitansState] = {}
        for stream_id, raw_state in raw_states.items():
            if not isinstance(raw_state, Mapping):
                raise ValueError("invalid stream-state checkpoint entry")
            restored[str(stream_id)] = TitansState.from_state_dict(raw_state)
        self.stream_states = restored
