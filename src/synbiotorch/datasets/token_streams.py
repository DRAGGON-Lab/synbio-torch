"""Immutable, memory-mapped token streams for long-sequence training."""

from __future__ import annotations

import hashlib
import json
import operator
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from synbiotorch.tokenize.base import SpannedTokenizer
from synbiotorch.types import Design

TOKEN_DTYPE = np.dtype("<i4")
BASE_COUNT_DTYPE = np.dtype("<u4")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_values(values: Sequence[int], *, stream_id: str, field: str) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"stream {stream_id!r} has a non-integer {field}")
        try:
            result.append(operator.index(value))
        except TypeError as error:
            raise ValueError(f"stream {stream_id!r} has a non-integer {field}") from error
    return result


@dataclass(frozen=True)
class TokenStreamRecord:
    """Location and aggregate accounting for one logical token stream."""

    stream_id: str
    split: str
    shard_index: int
    token_offset: int
    token_count: int
    represented_base_count: int


@dataclass(frozen=True)
class TokenStream:
    """A logical stream backed by either tuples or memory-mapped array views."""

    stream_id: str
    input_ids: Sequence[int]
    base_counts: Sequence[int]

    @property
    def examples(self) -> int:
        return max(0, len(self.input_ids) - 1)


@dataclass(frozen=True)
class EncodedTokenStream:
    """Domain-neutral input to token-stream materialization."""

    stream_id: str
    input_ids: Sequence[int]
    represented_base_counts: Sequence[int]
    split: str = "train"


def _tokenizer_metadata(tokenizer: SpannedTokenizer) -> dict[str, object]:
    kind = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    metadata: dict[str, object] = {
        "kind": kind,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
    }
    for name in ("k", "stride"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            metadata[name] = value
    return metadata


def materialize_token_streams(
    designs: Iterable[Design],
    tokenizer: SpannedTokenizer,
    output_dir: str | Path,
    *,
    tokens_per_shard: int = 4_000_000,
    source_fingerprint: str | None = None,
) -> "TokenStreamStore":
    """Tokenize design records into the generic immutable stream store."""

    def encoded_streams() -> Iterable[EncodedTokenStream]:
        for design in designs:
            if design.sequence is None or not design.sequence.elements:
                continue
            encoded = tokenizer.tokenize_content_with_spans(design.sequence.elements)
            split = design.raw.get("split", "train")
            if not isinstance(split, str) or not split:
                raise ValueError(f"stream {design.iri!r} has an invalid split")
            yield EncodedTokenStream(
                stream_id=design.iri,
                input_ids=encoded.input_ids,
                represented_base_counts=encoded.represented_base_counts,
                split=split,
            )

    return materialize_encoded_streams(
        encoded_streams(),
        output_dir,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer_metadata=_tokenizer_metadata(tokenizer),
        tokens_per_shard=tokens_per_shard,
        source_fingerprint=source_fingerprint,
    )


def materialize_encoded_streams(
    streams: Iterable[EncodedTokenStream],
    output_dir: str | Path,
    *,
    pad_token_id: int,
    tokenizer_metadata: dict[str, object] | None = None,
    tokens_per_shard: int = 4_000_000,
    source_fingerprint: str | None = None,
) -> "TokenStreamStore":
    """Publish pre-tokenized logical streams as an immutable mmap store.

    A logical stream is never split between shards. ``tokens_per_shard`` is a
    soft target: a single larger stream receives its own oversized shard.
    """

    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be positive")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"token stream store already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    staging.mkdir()

    records: list[TokenStreamRecord] = []
    shards: list[dict[str, object]] = []
    shard_tokens: list[int] = []
    shard_base_counts: list[int] = []
    stream_ids: set[str] = set()

    def flush_shard() -> None:
        if not shard_tokens:
            return
        shard_index = len(shards)
        token_name = f"tokens_{shard_index:05d}.npy"
        base_name = f"base_counts_{shard_index:05d}.npy"
        token_path = staging / token_name
        base_path = staging / base_name
        np.save(token_path, np.asarray(shard_tokens, dtype=TOKEN_DTYPE), allow_pickle=False)
        np.save(base_path, np.asarray(shard_base_counts, dtype=BASE_COUNT_DTYPE), allow_pickle=False)
        shards.append(
            {
                "index": shard_index,
                "token_file": token_name,
                "token_sha256": _sha256(token_path),
                "base_count_file": base_name,
                "base_count_sha256": _sha256(base_path),
                "token_count": len(shard_tokens),
            }
        )
        shard_tokens.clear()
        shard_base_counts.clear()

    try:
        for stream in streams:
            if not isinstance(stream.stream_id, str) or not stream.stream_id:
                raise ValueError("stream identity must be a non-empty string")
            if not isinstance(stream.split, str) or not stream.split:
                raise ValueError(f"stream {stream.stream_id!r} has an invalid split")
            token_values = _integer_values(stream.input_ids, stream_id=stream.stream_id, field="token id")
            base_values = _integer_values(
                stream.represented_base_counts,
                stream_id=stream.stream_id,
                field="represented-unit count",
            )
            if len(token_values) != len(base_values):
                raise ValueError(f"stream {stream.stream_id!r} has mismatched token and represented-unit counts")
            if len(token_values) < 2:
                continue
            if stream.stream_id in stream_ids:
                raise ValueError(f"duplicate stream identity: {stream.stream_id}")
            if shard_tokens and len(shard_tokens) + len(token_values) > tokens_per_shard:
                flush_shard()
            if any(token < 0 or token > np.iinfo(TOKEN_DTYPE).max for token in token_values):
                raise ValueError(f"stream {stream.stream_id!r} has a token outside the int32 range")
            if any(count <= 0 or count > np.iinfo(BASE_COUNT_DTYPE).max for count in base_values):
                raise ValueError(f"stream {stream.stream_id!r} has an invalid represented-unit count")
            stream_ids.add(stream.stream_id)
            records.append(
                TokenStreamRecord(
                    stream_id=stream.stream_id,
                    split=stream.split,
                    shard_index=len(shards),
                    token_offset=len(shard_tokens),
                    token_count=len(token_values),
                    represented_base_count=sum(base_values),
                )
            )
            shard_tokens.extend(token_values)
            shard_base_counts.extend(base_values)
        flush_shard()
        if not records:
            raise ValueError("token stream materialization requires at least one two-token sequence")

        index_path = staging / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
        manifest = {
            "format": "synbiotorch-token-streams",
            "format_version": 1,
            "token_dtype": TOKEN_DTYPE.str,
            "base_count_dtype": BASE_COUNT_DTYPE.str,
            "tokenizer": {**(tokenizer_metadata or {}), "pad_token_id": pad_token_id},
            "source_fingerprint": source_fingerprint,
            "stream_count": len(records),
            "token_count": sum(record.token_count for record in records),
            "represented_base_count": sum(record.represented_base_count for record in records),
            "index_file": index_path.name,
            "index_sha256": _sha256(index_path),
            "shards": shards,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return TokenStreamStore(target)


class TokenStreamStore:
    """Read-only view over a materialized token-stream manifest."""

    def __init__(self, path: str | Path, *, verify_checksums: bool = True) -> None:
        self.path = Path(path)
        with (self.path / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("format") != "synbiotorch-token-streams" or manifest.get("format_version") != 1:
            raise ValueError("unsupported token stream manifest")
        if manifest.get("token_dtype") != TOKEN_DTYPE.str:
            raise ValueError("token stream token dtype is incompatible")
        if manifest.get("base_count_dtype") != BASE_COUNT_DTYPE.str:
            raise ValueError("token stream base-count dtype is incompatible")
        if not isinstance(manifest.get("shards"), list) or not isinstance(manifest.get("tokenizer"), dict):
            raise ValueError("token stream manifest structure is invalid")
        self.manifest: dict[str, object] = manifest
        index_path = self.path / str(manifest["index_file"])
        if verify_checksums and _sha256(index_path) != manifest.get("index_sha256"):
            raise ValueError("token stream index checksum mismatch")
        with index_path.open(encoding="utf-8") as handle:
            self.records = tuple(TokenStreamRecord(**json.loads(line)) for line in handle if line.strip())
        if len(self.records) != manifest.get("stream_count"):
            raise ValueError("token stream manifest count does not match its index")
        self._shards = cast(list[dict[str, object]], manifest["shards"])
        self._arrays: dict[int, tuple[NDArray[np.int32], NDArray[np.uint32]]] = {}
        if verify_checksums:
            self.verify_checksums()
        self._validate_index()

    @property
    def pad_token_id(self) -> int:
        tokenizer = cast(dict[str, object], self.manifest["tokenizer"])
        value = tokenizer.get("pad_token_id")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("token stream manifest has an invalid pad token id")
        return value

    def verify_checksums(self) -> None:
        """Read every data file and reject corruption before training."""

        for shard in self._shards:
            token_path = self.path / str(shard["token_file"])
            base_path = self.path / str(shard["base_count_file"])
            if _sha256(token_path) != shard.get("token_sha256"):
                raise ValueError(f"token stream shard checksum mismatch: {token_path.name}")
            if _sha256(base_path) != shard.get("base_count_sha256"):
                raise ValueError(f"token stream base-count checksum mismatch: {base_path.name}")

    def _arrays_for(self, shard_index: int) -> tuple[NDArray[np.int32], NDArray[np.uint32]]:
        arrays = self._arrays.get(shard_index)
        if arrays is None:
            try:
                shard = self._shards[shard_index]
            except IndexError as error:
                raise ValueError(f"token stream references unknown shard {shard_index}") from error
            tokens = np.load(self.path / str(shard["token_file"]), mmap_mode="r", allow_pickle=False)
            base_counts = np.load(self.path / str(shard["base_count_file"]), mmap_mode="r", allow_pickle=False)
            if tokens.dtype != TOKEN_DTYPE or base_counts.dtype != BASE_COUNT_DTYPE:
                raise ValueError(f"token stream shard {shard_index} has incompatible dtypes")
            if tokens.ndim != 1 or base_counts.ndim != 1 or tokens.shape != base_counts.shape:
                raise ValueError(f"token stream shard {shard_index} has incompatible shapes")
            token_count = shard.get("token_count")
            if not isinstance(token_count, int) or isinstance(token_count, bool):
                raise ValueError(f"token stream shard {shard_index} has an invalid token count")
            if len(tokens) != token_count:
                raise ValueError(f"token stream shard {shard_index} has the wrong token count")
            arrays = (tokens, base_counts)
            self._arrays[shard_index] = arrays
        return arrays

    def _validate_index(self) -> None:
        stream_ids: set[str] = set()
        total_tokens = 0
        total_bases = 0
        for record in self.records:
            if record.shard_index < 0:
                raise ValueError(f"invalid shard index for stream {record.stream_id!r}")
            if record.stream_id in stream_ids:
                raise ValueError(f"duplicate stream identity in index: {record.stream_id}")
            stream_ids.add(record.stream_id)
            tokens, base_counts = self._arrays_for(record.shard_index)
            start = record.token_offset
            stop = start + record.token_count
            if start < 0 or record.token_count < 2 or stop > len(tokens):
                raise ValueError(f"invalid token range for stream {record.stream_id!r}")
            represented = int(base_counts[start:stop].sum(dtype=np.uint64))
            if represented != record.represented_base_count:
                raise ValueError(f"represented-base count mismatch for stream {record.stream_id!r}")
            total_tokens += record.token_count
            total_bases += represented
        if total_tokens != self.manifest.get("token_count"):
            raise ValueError("token stream manifest token count is inconsistent")
        if total_bases != self.manifest.get("represented_base_count"):
            raise ValueError("token stream manifest represented-base count is inconsistent")

    def streams(
        self,
        *,
        split: str | None = None,
        owner_rank: int = 0,
        owner_count: int = 1,
    ) -> tuple[TokenStream, ...]:
        """Return deterministic, disjoint stream ownership for one rank.

        Ownership is assigned after split filtering by immutable index position.
        Therefore all ranks must open the same manifest with the same split.
        """

        if owner_count <= 0 or not 0 <= owner_rank < owner_count:
            raise ValueError("owner_rank must be in [0, owner_count)")
        selected = (record for record in self.records if split is None or record.split == split)
        streams: list[TokenStream] = []
        for split_index, record in enumerate(selected):
            if split_index % owner_count != owner_rank:
                continue
            tokens, base_counts = self._arrays_for(record.shard_index)
            start = record.token_offset
            stop = start + record.token_count
            streams.append(
                TokenStream(
                    stream_id=record.stream_id,
                    input_ids=cast(Sequence[int], tokens[start:stop]),
                    base_counts=cast(Sequence[int], base_counts[start:stop]),
                )
            )
        return tuple(streams)
