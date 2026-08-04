"""Run configuration.

A single Pydantic-validated object fully specifies a run. Every field has a
sensible default so configs stay small, and the whole resolved object is
serialized into the run output directory for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from synbiotorch.exceptions import ConfigError


class CorpusConfig(BaseModel):
    source: Literal["fasta", "sbol", "genbank", "table", "synthetic", "sbol_db"] = "synthetic"

    # sbol_db source
    base_url: str | None = None
    username: str | None = None
    password: str | None = None
    record_class: str | None = None
    role: str | None = None
    document_id: str | None = None

    # file sources (fasta / sbol / genbank)
    path: str | None = None
    # Roots identities for GenBank import and the SBOL2->3 upgrade fallback.
    namespace: str | None = None
    # FASTA/table alphabet; "auto" detects DNA/RNA/protein from sequence content.
    alphabet: Literal["auto", "dna", "rna", "protein"] = "auto"

    # table source (CSV/TSV): which columns hold the sequence, label, and id.
    sequence_column: str | None = None
    label_column: str | None = None
    id_column: str | None = None

    # synthetic source (in-memory fixture generator, for development/testing)
    n: int = 64
    synthetic_seed: int = 0

    # Where to read the supervised label from. For sbol_db this is a predicate
    # local-name looked up in the object's `data` slice; for FASTA it is parsed
    # from the header; for SBOL files it is an annotation predicate local-name.
    # None means unlabeled (pretraining).
    label_key: str | None = None

    cache_dir: str = ".synbiotorch_cache"
    # Rows per Parquet shard when materializing. Sharding keeps both writing and
    # streaming memory-bounded over a corpus larger than RAM.
    shard_size: int = 50_000

    @model_validator(mode="after")
    def _check_source(self) -> "CorpusConfig":
        if self.source == "sbol_db" and not self.base_url:
            raise ConfigError("corpus.base_url is required when source is 'sbol_db'")
        if self.source in ("fasta", "sbol", "genbank", "table") and not self.path:
            raise ConfigError(f"corpus.path is required when source is '{self.source}'")
        if self.source == "genbank" and not self.namespace:
            raise ConfigError("corpus.namespace is required when source is 'genbank'")
        if self.source == "table" and not self.sequence_column:
            raise ConfigError("corpus.sequence_column is required when source is 'table'")
        return self


class TokenizerConfig(BaseModel):
    kind: Literal["hf", "kmer", "nonoverlap_kmer", "char"] = "hf"
    k: int = 6
    stride: int = 1
    max_length: int = 512
    model_name: str = "zhihan1996/DNABERT-2-117M"
    # char tokenizer alphabet: nucleotide or protein.
    alphabet: Literal["dna", "protein"] = "dna"


class EncoderConfig(BaseModel):
    kind: Literal["sequence", "structure_aware", "graph"] = "sequence"
    # structure_aware: role IRIs that get dedicated boundary markers (None = a
    # default SO set); whether to mark reverse-complement features.
    roles: tuple[str, ...] | None = None
    mark_orientation: bool = True


class ArchConfig(BaseModel):
    """Architecture for a from-scratch model (used when ``model.from_scratch``).

    ``model_type`` selects any HuggingFace architecture: ``bert``/``modernbert``
    for MLM, ``gpt2``/``gpt_neox``/``llama`` for causal. RoPE-based types
    (``modernbert``, ``gpt_neox``, ``llama``) extrapolate past
    ``max_position_embeddings`` instead of failing, which is what makes long
    context practical; ``rope_theta`` tunes their rotary base.
    """

    model_type: str = "bert"
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_position_embeddings: int = 2048
    # PyTorch SDPA dispatches to FlashAttention kernels on CUDA and works on CPU;
    # ``flash_attention_2`` needs the flash-attn package + CUDA.
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    # Rotary base for RoPE architectures; None uses the architecture's default.
    rope_theta: float | None = None


class TitansConfig(BaseModel):
    """Paper-deep TITANS Memory-as-Context architecture."""

    num_layers: int = 4
    num_attention_heads: int = 4
    persistent_tokens: int = 4
    segment_length: int = 32
    memory_expansion_factor: int = 4
    projection_kernel_size: int = 4
    alpha_initial: float = 0.001
    eta_initial: float = 0.9
    theta_initial: float = 0.001
    tie_embeddings: bool = True

    @model_validator(mode="after")
    def _check_titans(self) -> "TitansConfig":
        if self.num_layers <= 0 or self.num_attention_heads <= 0 or self.persistent_tokens <= 0:
            raise ConfigError("TITANS layers, heads, and persistent tokens must be positive")
        if self.segment_length != 32:
            raise ConfigError("the paper-deep TITANS MAC implementation requires segment_length: 32")
        if self.memory_expansion_factor <= 0:
            raise ConfigError("TITANS memory expansion must be positive")
        if self.projection_kernel_size < 2:
            raise ConfigError("TITANS projection kernel must be at least two")
        if not 0.0 < self.alpha_initial < 1.0:
            raise ConfigError("TITANS alpha_initial must be in (0, 1)")
        if not 0.0 < self.eta_initial < 1.0:
            raise ConfigError("TITANS eta_initial must be in (0, 1)")
        if not 0.0 < self.theta_initial < 1.0:
            raise ConfigError("TITANS theta_initial must be in (0, 1)")
        return self


class ModelConfig(BaseModel):
    kind: Literal["transformers", "titans_mac"] = "transformers"
    # A HuggingFace hub id, or a local path to a checkpoint saved by a prior run
    # (e.g. the backbone written out by an MLM pretraining run).
    backbone: str = "zhihan1996/DNABERT-2-117M"
    hidden_size: int = 768
    dropout: float = 0.1
    # When true, build the encoder from ``arch`` + the tokenizer vocab instead of
    # downloading pretrained weights — the from-scratch MLM pretraining path.
    from_scratch: bool = False
    arch: ArchConfig = Field(default_factory=ArchConfig)
    titans: TitansConfig = Field(default_factory=TitansConfig)

    @model_validator(mode="after")
    def _check_model(self) -> "ModelConfig":
        if self.kind == "titans_mac" and self.hidden_size % self.titans.num_attention_heads:
            raise ConfigError("model.hidden_size must be divisible by TITANS attention heads")
        return self


class TaskConfig(BaseModel):
    kind: Literal["supervised", "mlm", "frozen", "causal"] = "frozen"
    objective: Literal["regression", "classification"] = "regression"
    num_classes: int | None = None
    target_transform: Literal["none", "log1p"] = "none"
    mlm_probability: float = 0.15

    @model_validator(mode="after")
    def _check_classes(self) -> "TaskConfig":
        if self.objective == "classification" and not self.num_classes:
            raise ConfigError("task.num_classes is required for classification")
        return self


class EarlyStopConfig(BaseModel):
    monitor: str = "val_loss"
    patience: int = 5
    mode: Literal["min", "max"] = "min"
    min_delta: float = 0.0


class SplitConfig(BaseModel):
    # ``hash`` assigns each record to a partition by hashing its IRI, so the split
    # needs no global index and is stable as the corpus grows — required for the
    # streaming path. ``random``/``stratified`` are the in-memory index splits.
    # ``column`` honors a partition the source already carries (a ``split`` field
    # per record valued ``train``/``val``/``test``), for datasets that ship a
    # fixed, published split; ``ratios`` are unused in that case.
    strategy: Literal["random", "stratified", "hash", "column"] = "random"
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    # Raw-field key read when ``strategy == 'column'``.
    column: str = "split"

    @model_validator(mode="after")
    def _check_ratios(self) -> "SplitConfig":
        if abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ConfigError(f"splits.ratios must sum to 1.0, got {self.ratios}")
        return self


class WandbConfig(BaseModel):
    """Weights & Biases experiment tracking. Disabled by default.

    The API key is read from the ``WANDB_API_KEY`` environment variable and is
    never a config field. The resolved ``RunConfig`` is logged as the W&B run
    config, so the dashboard inherits the same reproducibility anchor as the
    serialized ``config.resolved.yaml``.
    """

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "online"
    run_name: str | None = None
    tags: tuple[str, ...] = ()
    group: str | None = None
    job_type: str | None = None
    # Log gradient/parameter histograms via wandb.watch. Heavy on large
    # backbones, so opt-in.
    watch_model: bool = False
    # Steps between per-step metric flushes and gradient logs.
    log_freq: int = 100
    # Push the best checkpoint as a model Artifact at train end.
    log_model: bool = True


class DistributedConfig(BaseModel):
    """Multi-GPU / multi-node strategy. ``none`` is plain single-process.

    ``ddp`` replicates the model and all-reduces gradients across ranks (works on
    CPU/gloo, so the orchestration is testable without GPUs). It does not shard
    parameters, so it gives no memory saving — training a model larger than one
    device (e.g. via FSDP) is a future addition that needs GPU validation.
    """

    strategy: Literal["none", "ddp"] = "none"
    find_unused_parameters: bool = False


class PackingConfig(BaseModel):
    """Token packing for language-model pretraining: concatenate tokenized
    documents into fixed-length ``block_size`` blocks with no padding."""

    enabled: bool = False
    block_size: int = 512


class StreamConfig(BaseModel):
    """Ordered stateful sequence execution for recurrent or fast-weight models."""

    enabled: bool = False
    gradient_horizon: int = 3
    memory_mode: Literal["adaptive", "frozen", "none"] = "adaptive"
    shuffle_streams: bool = True

    @model_validator(mode="after")
    def _check_stream(self) -> "StreamConfig":
        if self.gradient_horizon <= 0:
            raise ConfigError("stream.gradient_horizon must be positive")
        return self


class TrainConfig(BaseModel):
    batch_size: int = 16
    epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    amp: bool = True
    # Autocast precision when ``amp`` is on and the device is CUDA. ``bf16`` needs
    # no loss scaler and is the stable choice for large models; ``fp32`` disables
    # autocast even when ``amp`` is on.
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    num_workers: int = 0
    # Step-budgeted training. When ``max_steps`` is set it, not ``epochs``, ends
    # the run, and evaluation/checkpointing can run on a step cadence rather than
    # at epoch boundaries — the right mode for pretraining over a large corpus.
    max_steps: int | None = None
    eval_every_n_steps: int | None = None
    checkpoint_every_n_steps: int | None = None
    gradient_checkpointing: bool = False
    compile: bool = False
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    early_stop: EarlyStopConfig | None = None


class RunConfig(BaseModel):
    seed: int = 42
    output_dir: str = "runs/default"
    # Stream the corpus from sharded Parquet instead of loading it all into RAM.
    # Requires a hash split and a step budget (``train.max_steps``). Sequence and
    # MLM modalities only; the graph path stays in-memory.
    streaming: bool = False
    corpus: CorpusConfig
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    packing: PackingConfig = Field(default_factory=PackingConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def _check_cross_layer_contracts(self) -> "RunConfig":
        if self.model.kind != "titans_mac":
            if self.stream.enabled:
                raise ConfigError("stream.enabled currently requires model.kind: titans_mac")
            return self
        if self.task.kind != "causal":
            raise ConfigError("model.kind: titans_mac requires task.kind: causal")
        if self.tokenizer.kind != "nonoverlap_kmer":
            raise ConfigError("model.kind: titans_mac currently requires tokenizer.kind: nonoverlap_kmer")
        if not self.stream.enabled:
            raise ConfigError("model.kind: titans_mac requires stream.enabled: true")
        if self.streaming or self.packing.enabled:
            raise ConfigError("TITANS ordered streams do not use generic streaming or token packing")
        if self.train.max_steps is None:
            raise ConfigError("TITANS ordered-stream training requires train.max_steps")
        if self.train.num_workers != 0:
            raise ConfigError("TITANS stateful cursor checkpointing currently requires train.num_workers: 0")
        if self.train.distributed.strategy != "none":
            raise ConfigError("distributed TITANS stateful forward and per-rank checkpointing are not implemented")
        if self.train.amp or self.train.precision != "fp32":
            raise ConfigError("adaptive TITANS AMP is not yet qualified; use train.amp: false and precision: fp32")
        if self.train.compile or self.train.gradient_checkpointing:
            raise ConfigError("adaptive TITANS compile and gradient checkpointing are not yet supported")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        text = Path(path).read_text()
        raw = yaml.safe_load(text) or {}
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
