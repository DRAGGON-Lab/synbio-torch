"""Deterministic, domain-neutral systems probe for stateful TITANS training.

This exercises mmap streams, represented-unit metrics, stateful window
ownership, higher-order memory updates, validation, and exact stream boundaries
without requiring a production corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from synbiotorch.config import TitansConfig, TrainConfig
from synbiotorch.datasets import EncodedTokenStream, OrderedStreamWindowDataset, materialize_encoded_streams
from synbiotorch.engine.stateful import StatefulCausalBatchAdapter
from synbiotorch.engine.trainer import Trainer
from synbiotorch.models.titans import TitansCausalLM
from synbiotorch.reproducibility import set_seed
from synbiotorch.tasks.causal import CausalLMTask


class CountingStatefulAdapter(StatefulCausalBatchAdapter):
    """Count the exact valid tokens and represented units observed by a run."""

    def __init__(self) -> None:
        super().__init__()
        self.observed_tokens = 0
        self.observed_units = 0

    def metric_contributions(
        self,
        logits: Tensor,
        labels: Tensor,
        batch: dict[str, Tensor | tuple[str, ...]],
    ) -> dict[str, tuple[Tensor, Tensor]]:
        contributions = super().metric_contributions(logits, labels, batch)
        self.observed_tokens += int(contributions["loss"][1].detach().cpu())
        self.observed_units += int(contributions["bits_per_base"][1].detach().cpu())
        return contributions


def deterministic_streams(
    *,
    train_streams: int,
    validation_streams: int,
    tokens_per_stream: int,
    vocab_size: int,
    seed: int,
) -> list[EncodedTokenStream]:
    """Create reproducible stateful streams with non-uniform metric weights."""

    if train_streams <= 0 or validation_streams <= 0 or tokens_per_stream < 2 or vocab_size < 4:
        raise ValueError("stream counts must be positive, token count >= 2, and vocab size >= 4")
    streams: list[EncodedTokenStream] = []
    for index in range(train_streams + validation_streams):
        generator = torch.Generator().manual_seed(seed + index)
        first = int(torch.randint(1, vocab_size, (), generator=generator))
        tokens = [first]
        for position in range(1, tokens_per_stream):
            tokens.append(1 + (tokens[-1] * 5 + index + position % 7) % (vocab_size - 1))
        weights = [1 + ((position + index) % 4) for position in range(tokens_per_stream)]
        split = "train" if index < train_streams else "val"
        streams.append(
            EncodedTokenStream(
                stream_id=f"synthetic:{split}:{index}",
                input_ids=tokens,
                represented_base_counts=weights,
                split=split,
            )
        )
    return streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/titans-systems-probe"))
    parser.add_argument("--train-streams", type=int, default=6)
    parser.add_argument("--validation-streams", type=int, default=2)
    parser.add_argument("--tokens-per-stream", type=int, default=1024)
    parser.add_argument("--tokens-per-shard", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_size = 32
    probe_spec: dict[str, Any] = {
        "train_streams": args.train_streams,
        "validation_streams": args.validation_streams,
        "tokens_per_stream": args.tokens_per_stream,
        "vocab_size": vocab_size,
        "seed": args.seed,
    }
    fingerprint = hashlib.sha256(json.dumps(probe_spec, sort_keys=True).encode()).hexdigest()[:16]
    store_path = args.output_dir / f"streams-{fingerprint}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if store_path.exists():
        from synbiotorch.datasets import TokenStreamStore

        store = TokenStreamStore(store_path)
        if store.manifest.get("source_fingerprint") != fingerprint:
            raise ValueError("existing probe store does not match the requested specification")
    else:
        store = materialize_encoded_streams(
            deterministic_streams(**probe_spec),
            store_path,
            pad_token_id=0,
            tokenizer_metadata={"kind": "deterministic-systems-probe", "vocab_size": vocab_size},
            tokens_per_shard=args.tokens_per_shard,
            source_fingerprint=fingerprint,
        )

    window_options = {
        "batch_size": args.batch_size,
        "segment_length": 32,
        "gradient_horizon": 1,
        "seed": args.seed,
        "shuffle_streams": False,
    }
    train_data = OrderedStreamWindowDataset.from_store(store, split="train", **window_options)
    validation_data = OrderedStreamWindowDataset.from_store(store, split="val", **window_options)
    train_adapter = CountingStatefulAdapter()
    validation_adapter = CountingStatefulAdapter()
    set_seed(args.seed)
    model = TitansCausalLM(
        TitansConfig(
            num_layers=1,
            num_attention_heads=2,
            persistent_tokens=2,
            memory_expansion_factor=2,
        ),
        vocab_size=vocab_size,
        pad_token_id=0,
        d_model=8,
    )
    trainer = Trainer(
        model,
        CausalLMTask(),
        TrainConfig(batch_size=args.batch_size, max_steps=args.steps, amp=False, precision="fp32"),
        batch_adapter=train_adapter,
        val_batch_adapter=validation_adapter,
        device=torch.device("cpu"),
    )
    started = time.perf_counter()
    metrics = trainer.fit(
        DataLoader(train_data, batch_size=None, num_workers=0),
        DataLoader(validation_data, batch_size=None, num_workers=0),
    )
    elapsed = time.perf_counter() - started
    report = {
        "probe": probe_spec,
        "store": {
            "path": str(store.path),
            "shards": len(store.manifest["shards"]),
            "tokens": store.manifest["token_count"],
            "represented_units": store.manifest["represented_base_count"],
        },
        "run": {
            "steps": trainer.global_step,
            "seconds_including_validation": elapsed,
            "observed_train_tokens": train_adapter.observed_tokens,
            "observed_train_units": train_adapter.observed_units,
            "train_tokens_per_second_including_validation": train_adapter.observed_tokens / elapsed,
        },
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
