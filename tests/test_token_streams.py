from __future__ import annotations

import json

import numpy as np
import pytest

from synbiotorch.datasets import OrderedStreamWindowDataset
from synbiotorch.datasets.token_streams import (
    EncodedTokenStream,
    TokenStreamStore,
    materialize_encoded_streams,
    materialize_token_streams,
)
from synbiotorch.tokenize.kmer import NonOverlappingKmerTokenizer
from synbiotorch.types import Design, Sequence


def _design(name: str, content: str, *, split: str = "train") -> Design:
    return Design(
        iri=f"stream:{name}",
        record_class="sequence",
        sequence=Sequence(content),
        raw={"split": split},
    )


def test_token_stream_store_is_mmap_backed_and_preserves_accounting(tmp_path):
    tokenizer = NonOverlappingKmerTokenizer(k=2)
    store = materialize_token_streams(
        [
            _design("a", "ACGTAC"),
            _design("b", "TTTTN"),
            _design("validation", "GGGG", split="val"),
        ],
        tokenizer,
        tmp_path / "streams",
        tokens_per_shard=5,
        source_fingerprint="fixture-v1",
    )

    assert store.manifest["stream_count"] == 3
    assert store.manifest["represented_base_count"] == 15
    assert store.manifest["source_fingerprint"] == "fixture-v1"
    assert len(store.manifest["shards"]) == 2
    train = store.streams(split="train")
    assert [stream.stream_id for stream in train] == ["stream:a", "stream:b"]
    assert isinstance(train[0].input_ids, np.memmap)
    assert sum(int(value) for stream in train for value in stream.base_counts) == 11

    reopened = TokenStreamStore(tmp_path / "streams")
    assert [list(stream.input_ids) for stream in reopened.streams(split="val")] == [tokenizer.tokenize_content("GGGG")]


def test_token_stream_ownership_is_disjoint_deterministic_and_complete(tmp_path):
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    expected = [f"stream:{index}" for index in range(7)]
    store = materialize_token_streams(
        [_design(str(index), "ACGT") for index in range(7)],
        tokenizer,
        tmp_path / "streams",
    )

    owners = [
        tuple(stream.stream_id for stream in store.streams(split="train", owner_rank=rank, owner_count=3))
        for rank in range(3)
    ]
    assert owners == [tuple(expected[0::3]), tuple(expected[1::3]), tuple(expected[2::3])]
    assert set().union(*(set(owner) for owner in owners)) == set(expected)
    assert sum(len(owner) for owner in owners) == len(expected)
    assert owners[1] == tuple(stream.stream_id for stream in store.streams(split="train", owner_rank=1, owner_count=3))


def test_pretokenized_streams_are_domain_neutral(tmp_path):
    store = materialize_encoded_streams(
        [
            EncodedTokenStream(
                stream_id="document:alpha",
                input_ids=(5, 8, 13),
                represented_base_counts=(2, 1, 4),
                split="evaluation",
            )
        ],
        tmp_path / "streams",
        pad_token_id=0,
        tokenizer_metadata={"kind": "fixture"},
    )

    stream = store.streams(split="evaluation")[0]
    assert list(stream.input_ids) == [5, 8, 13]
    assert list(stream.base_counts) == [2, 1, 4]
    assert store.manifest["represented_base_count"] == 7


def test_mmap_windows_never_cross_stream_boundaries(tmp_path):
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    store = materialize_token_streams(
        [_design("a", "AAAA"), _design("c", "CCCC")],
        tokenizer,
        tmp_path / "streams",
    )
    dataset = OrderedStreamWindowDataset.from_store(
        store,
        split="train",
        batch_size=2,
        segment_length=8,
        gradient_horizon=1,
        shuffle_streams=False,
    )

    batch = next(iter(dataset))
    valid = batch["valid_mask"]
    assert batch["stream_ids"] == ("stream:a", "stream:c")
    assert batch["end_of_stream"].tolist() == [True, True]
    assert batch["labels"][0][valid[0]].tolist() == tokenizer.tokenize_content("AAA")
    assert batch["labels"][1][valid[1]].tolist() == tokenizer.tokenize_content("CCC")
    assert batch["represented_base_counts"][valid].sum().item() == 6


def test_mmap_window_cursor_resumes_at_the_exact_next_token(tmp_path):
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    store = materialize_token_streams([_design("long", "ACGT" * 4)], tokenizer, tmp_path / "streams")
    dataset = OrderedStreamWindowDataset.from_store(
        store,
        batch_size=1,
        segment_length=4,
        gradient_horizon=1,
        shuffle_streams=False,
    )
    iterator = iter(dataset)
    next(iterator)
    state = dataset.state_dict()
    expected = next(iterator)

    restored = OrderedStreamWindowDataset.from_store(
        TokenStreamStore(tmp_path / "streams"),
        batch_size=1,
        segment_length=4,
        gradient_horizon=1,
        shuffle_streams=False,
    )
    restored.load_state_dict(state)
    actual = next(iter(restored))
    assert actual["stream_ids"] == expected["stream_ids"]
    for name in ("input_ids", "labels", "valid_mask", "represented_base_counts"):
        assert np.array_equal(actual[name].numpy(), expected[name].numpy())


def test_stateful_window_scheduler_rejects_worker_local_cursor_copies(tmp_path, monkeypatch):
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    store = materialize_token_streams([_design("a", "ACGT")], tokenizer, tmp_path / "streams")
    dataset = OrderedStreamWindowDataset.from_store(store, batch_size=1, shuffle_streams=False)
    monkeypatch.setattr("synbiotorch.datasets.stream_windows.get_worker_info", lambda: object())

    with pytest.raises(RuntimeError, match="num_workers=0"):
        next(iter(dataset))


def test_token_stream_store_rejects_index_corruption_and_overwrite(tmp_path):
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    path = tmp_path / "streams"
    materialize_token_streams([_design("a", "ACGT")], tokenizer, path)
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_token_streams([_design("b", "TGCA")], tokenizer, path)

    index_path = path / "index.jsonl"
    record = json.loads(index_path.read_text().strip())
    record["token_count"] += 1
    index_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="index checksum mismatch"):
        TokenStreamStore(path)
