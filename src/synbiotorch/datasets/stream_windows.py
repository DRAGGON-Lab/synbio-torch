"""Ordered, checkpointable gradient-horizon windows for stateful causal models."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence, TypedDict

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from synbiotorch.datasets.mlm_collator import IGNORE_INDEX
from synbiotorch.tokenize.base import SpannedTokenizer
from synbiotorch.types import Design


@dataclass(frozen=True)
class _TokenStream:
    stream_id: str
    input_ids: tuple[int, ...]
    base_counts: tuple[int, ...]

    @property
    def examples(self) -> int:
        return max(0, len(self.input_ids) - 1)


StreamWindowBatch = dict[str, Tensor | tuple[str, ...]]


class _WindowRow(TypedDict):
    input_ids: Tensor
    labels: Tensor
    valid_mask: Tensor
    represented_base_counts: Tensor
    segment_count: int
    start_of_stream: bool
    end_of_stream: bool
    next_offset: int


class OrderedStreamWindowDataset(IterableDataset[StreamWindowBatch]):
    """Interleave logical streams while preserving order within every stream.

    Each yielded object is already one complete batch with shape ``[B, H, C]``.
    A row stays on one stream for the complete gradient horizon ``H``; replacement
    streams enter only in the next batch. Cursor advancement occurs before yield,
    so a checkpoint taken after the optimizer step resumes at the next window.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        designs: Sequence[Design],
        tokenizer: SpannedTokenizer,
        *,
        batch_size: int,
        segment_length: int = 32,
        gradient_horizon: int = 3,
        seed: int = 42,
        shuffle_streams: bool = True,
    ) -> None:
        super().__init__()
        if batch_size <= 0 or segment_length <= 0 or gradient_horizon <= 0:
            raise ValueError("batch size, segment length, and gradient horizon must be positive")
        streams: list[_TokenStream] = []
        for design in designs:
            if design.sequence is None or not design.sequence.elements:
                continue
            encoded = tokenizer.tokenize_content_with_spans(design.sequence.elements)
            if len(encoded.input_ids) < 2:
                continue
            streams.append(
                _TokenStream(
                    stream_id=design.iri,
                    input_ids=tuple(encoded.input_ids),
                    base_counts=tuple(encoded.represented_base_counts),
                )
            )
        if not streams:
            raise ValueError("ordered stream training requires at least one two-token sequence")
        if len({stream.stream_id for stream in streams}) != len(streams):
            raise ValueError("ordered stream identities must be unique")
        self.streams = tuple(streams)
        self.batch_size = batch_size
        self.pad_token_id = tokenizer.pad_token_id
        self.segment_length = segment_length
        self.gradient_horizon = gradient_horizon
        self.seed = seed
        self.shuffle_streams = shuffle_streams
        self.epoch = 0
        self.order: list[int] = []
        self.next_stream_cursor = 0
        self.active: list[tuple[int, int] | None] = []
        self.exhausted = False
        self.reset(epoch=0)

    def _fill(self, slot: int) -> None:
        self.active[slot] = None
        if self.next_stream_cursor < len(self.order):
            stream_index = self.order[self.next_stream_cursor]
            self.next_stream_cursor += 1
            self.active[slot] = (stream_index, 0)

    def reset(self, *, epoch: int | None = None) -> None:
        self.epoch = self.epoch + 1 if epoch is None else epoch
        self.order = list(range(len(self.streams)))
        if self.shuffle_streams:
            random.Random(self.seed + self.epoch).shuffle(self.order)
        self.next_stream_cursor = 0
        self.active = [None] * self.batch_size
        for slot in range(self.batch_size):
            self._fill(slot)
        self.exhausted = False

    def _window(self, stream: _TokenStream, offset: int) -> _WindowRow:
        width = self.gradient_horizon * self.segment_length
        count = min(width, stream.examples - offset)
        input_values = list(stream.input_ids[offset : offset + count])
        label_values = list(stream.input_ids[offset + 1 : offset + count + 1])
        base_values = list(stream.base_counts[offset + 1 : offset + count + 1])
        padding = width - count
        input_values.extend([self.pad_token_id] * padding)
        label_values.extend([IGNORE_INDEX] * padding)
        base_values.extend([0] * padding)
        valid_values = [True] * count + [False] * padding
        shape = (self.gradient_horizon, self.segment_length)
        return {
            "input_ids": torch.tensor(input_values, dtype=torch.long).view(shape),
            "labels": torch.tensor(label_values, dtype=torch.long).view(shape),
            "valid_mask": torch.tensor(valid_values, dtype=torch.bool).view(shape),
            "represented_base_counts": torch.tensor(base_values, dtype=torch.long).view(shape),
            "segment_count": math.ceil(count / self.segment_length),
            "start_of_stream": offset == 0,
            "end_of_stream": offset + count == stream.examples,
            "next_offset": offset + count,
        }

    def _next_batch(self) -> StreamWindowBatch:
        rows: list[_WindowRow] = []
        stream_ids: list[str] = []
        for slot, item in enumerate(self.active):
            if item is None:
                continue
            stream_index, offset = item
            stream = self.streams[stream_index]
            window = self._window(stream, offset)
            rows.append(window)
            stream_ids.append(stream.stream_id)
            if window["end_of_stream"]:
                self._fill(slot)
            else:
                self.active[slot] = (stream_index, window["next_offset"])
        if not rows:
            raise StopIteration
        self.exhausted = all(item is None for item in self.active)

        def stack(name: str) -> Tensor:
            values = [row[name] for row in rows]  # type: ignore[literal-required]
            if not all(isinstance(value, Tensor) for value in values):
                raise RuntimeError(f"stream window field {name!r} is not tensor-valued")
            return torch.stack(values)  # type: ignore[arg-type]

        return {
            "stream_ids": tuple(stream_ids),
            "input_ids": stack("input_ids"),
            "labels": stack("labels"),
            "valid_mask": stack("valid_mask"),
            "represented_base_counts": stack("represented_base_counts"),
            "segment_counts": torch.tensor([row["segment_count"] for row in rows], dtype=torch.long),
            "start_of_stream": torch.tensor([row["start_of_stream"] for row in rows], dtype=torch.bool),
            "end_of_stream": torch.tensor([row["end_of_stream"] for row in rows], dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[StreamWindowBatch]:
        if get_worker_info() is not None:
            raise RuntimeError("ordered stream windows require num_workers=0")
        if self.exhausted:
            self.reset()
        while not self.exhausted:
            yield self._next_batch()

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "stream_ids": [stream.stream_id for stream in self.streams],
            "batch_size": self.batch_size,
            "pad_token_id": self.pad_token_id,
            "segment_length": self.segment_length,
            "gradient_horizon": self.gradient_horizon,
            "seed": self.seed,
            "shuffle_streams": self.shuffle_streams,
            "epoch": self.epoch,
            "order": list(self.order),
            "next_stream_cursor": self.next_stream_cursor,
            "active": [list(item) if item is not None else None for item in self.active],
            "exhausted": self.exhausted,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        expected = {
            "format_version": self.FORMAT_VERSION,
            "stream_ids": [stream.stream_id for stream in self.streams],
            "batch_size": self.batch_size,
            "pad_token_id": self.pad_token_id,
            "segment_length": self.segment_length,
            "gradient_horizon": self.gradient_horizon,
            "seed": self.seed,
            "shuffle_streams": self.shuffle_streams,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"ordered stream dataset {name} changed across resume")
        order = payload.get("order")
        active = payload.get("active")
        if not isinstance(order, list) or sorted(map(int, order)) != list(range(len(self.streams))):
            raise ValueError("ordered stream dataset order is invalid")
        if not isinstance(active, list) or len(active) != self.batch_size:
            raise ValueError("ordered stream dataset active slots are invalid")
        restored: list[tuple[int, int] | None] = []
        for item in active:
            if item is None:
                restored.append(None)
                continue
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("ordered stream dataset cursor is invalid")
            stream_index, offset = map(int, item)
            if not 0 <= stream_index < len(self.streams):
                raise ValueError("ordered stream dataset references an unknown stream")
            if not 0 <= offset < self.streams[stream_index].examples:
                raise ValueError("ordered stream dataset token offset is invalid")
            restored.append((stream_index, offset))
        epoch = payload.get("epoch", 0)
        next_stream_cursor = payload.get("next_stream_cursor", 0)
        if not isinstance(epoch, int) or not isinstance(next_stream_cursor, int):
            raise ValueError("ordered stream dataset checkpoint counters are invalid")
        self.epoch = epoch
        self.order = list(map(int, order))
        self.next_stream_cursor = next_stream_cursor
        self.active = restored
        self.exhausted = bool(payload.get("exhausted", False))
