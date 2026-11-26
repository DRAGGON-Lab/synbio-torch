# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow semantic versioning.

## [Unreleased]

### Added

- **Model architecture is now a plug point.** `model.arch` is a tagged union on
  `family`, joining tokenizers, encoders, tasks, callbacks, batch modalities and
  data sources as a protocol-plus-factory axis. Adding a family is one config
  member, one `SequenceBackbone` subclass, and one branch in
  `build_architecture`.
- **`family: conv`** — a 1D convolutional motif scanner. Convolution plus
  pooling is position-invariant by construction, which suits regulatory DNA and
  needs far fewer parameters than attention.
- **Configurable pooling** (`arch.pooling`): masked `mean` (the previous fixed
  behavior), `max`, `cls`, and `flatten`, which preserves positional layout and
  sizes the head from the sequence length. Poolers hold no parameters, so
  existing checkpoints are unaffected.
- **`task.target_transform: standardize`** with `target_mean`/`target_std`, for
  heavy-tailed targets whose raw scale makes the squared-error gradient
  ill-conditioned.
- `Collator(pad_to=...)` for architectures whose head width depends on sequence
  length, so a record's features never depend on its batch-mates.

### Changed

- `build_model` takes `max_length` and sizes the head from the pooler's output
  width rather than from `model.hidden_size`.
- Fields that cannot apply to the chosen family (`from_scratch`, `backbone`,
  `hidden_size`) now raise `ConfigError` when set explicitly, instead of being
  silently ignored. `from_scratch` is implied for non-transformer families.
- `encoder.kind: graph` defaults `arch.family` to `graph`, so the graph model is
  described honestly in resolved configs instead of being tagged a transformer.

Existing configs and checkpoints are unaffected: an untagged `arch` resolves to
its previous meaning, and `state_dict` keys are unchanged.

## [0.2.0] - 2026-06-20

A rebrand to a general synthetic-biology ML library. **Breaking** throughout (the
library has no released users): the distribution and import names change, the core
record type is renamed, config keys change, and the external `sbol` CLI dependency
is replaced by an in-process native extension.

### Changed

- **Renamed `sbol-torch` → `synbio-torch`** (import `synbiotorch`, was `sboltorch`;
  CLI `synbiotorch`, was `sboltorch`; cache `.synbiotorch_cache`).
- **Core record `SbolObject` → `Design`** (`SbolSequence` → `Sequence`, field
  `sbol_class` → `record_class`). SBOL is now one source among many, not the
  framing.
- **Corpus sources moved to `synbiotorch.sources`**; `synbiotorch.data` keeps the
  source-neutral `Corpus` protocol, `build_corpus`, and materialization. The
  `local` source is replaced by explicit `fasta` / `sbol` / `genbank` sources.

### Added

- **Native sbol-rs parsing via PyO3.** GenBank import, SBOL 2→3 upgrade, and SBOL 3
  reading are bound in-process from the [sbol-rs](https://github.com/marpaia/sbol-rs)
  Rust crates (`synbiotorch._sbol`, built with maturin) — replacing the external
  `sbol` CLI shell-out and the `rdflib` dependency.
- **New data sources:** labeled CSV/TSV `table`, first-class `genbank`, and `sbol`
  (2 & 3); FASTA gains alphabet auto-detection.
- **Protein tokenization:** the `char` tokenizer takes `alphabet: dna | protein`.
- New example configs: `finetune_protein.yaml`, `benchmark_dna_classification.yaml`,
  `ingest_genbank.yaml`, with bundled CSV datasets.

### Removed

- The `rdflib` dependency, `scripts/normalize_sbol.sh`, and the `SBOL_BIN` CLI
  lookup — SBOL/GenBank parsing is now native and in-process.

## [0.1.1] - 2026-06-20

### Added

- **Causal-language-model pretraining and generation.** A `causal` objective
  (`task.kind: causal`) trains a decoder (`gpt2`, `gpt_neox`, `llama`, …) on
  next-token prediction. `synbiotorch generate` and `st.generate`/
  `st.generate_sequence` do autoregressive sampling (temperature / top-k / top-p)
  and design completion from a prefix. Tokenizers gained `decode`.
- **Streaming, sharded data.** Corpora materialize to sharded Parquet and can be
  streamed (`streaming: true`) so training no longer needs the corpus in RAM, with
  a stable `hash` split, multi-worker shard assignment, and optional token
  `packing` into fixed-length blocks for LM pretraining.
- **Long context & modern attention.** `model.arch` gained `attn_implementation`
  (SDPA by default — FlashAttention on CUDA) and `rope_theta`; RoPE architectures
  run past an absolute model's context limit.
- **Distributed training (DDP).** `train.distributed.strategy: ddp` replicates the
  model and all-reduces gradients across ranks (launch with `torchrun`), with
  rank-aware data sharding, rank-0-only checkpoints/logs, and cross-rank metric
  reduction. Data-parallel only; no parameter sharding.
- **Hardened training loop.** Resumable checkpoints (`synbiotorch train --resume`)
  carrying optimizer/scheduler/scaler/RNG state; step-budgeted training
  (`max_steps`, `eval_every_n_steps`, `checkpoint_every_n_steps`); `bf16`/`fp16`
  precision; gradient checkpointing; `torch.compile`.
- **sbol CLI normalization.** `scripts/normalize_sbol.sh` converts raw GenBank/
  SBOL2 inputs to SBOL3 via the `sbol` CLI for the `local` corpus source, with
  Component-centric SBOL3 parsing that unlocks the structure-aware and graph
  modalities on real data.
- Continuous integration via GitHub Actions.
- New example config `pretrain_causal_long.yaml` (RoPE decoder, SDPA, streamed +
  packed corpus).

### Changed

- `model.arch.max_position_embeddings` default raised from 1024 to 2048.
- Corpus materialization writes sharded Parquet (`corpus.shard_size`) instead of a
  single file; existing single-file caches are still read.
- Package now uses absolute imports throughout.

## [0.1.0]

- Initial release: SBOL/FASTA and sbol-db data sources, `sequence` /
  `structure_aware` / `graph` modalities, `supervised` / `frozen` / `mlm`
  objectives, `hf` / `kmer` / `char` tokenizers, a raw-PyTorch training engine with
  early stopping and AMP, reproducible Parquet caching, and Weights & Biases
  tracking.
