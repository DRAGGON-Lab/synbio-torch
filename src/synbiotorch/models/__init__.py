"""Model construction: assemble a backbone + head (or an MLM model) from config."""

from __future__ import annotations

import torch.nn as nn

from synbiotorch.config import ConvArch, GraphArch, ModelConfig, TaskConfig
from synbiotorch.exceptions import ConfigError
from synbiotorch.models.backbone import build_from_scratch_encoder, load_backbone
from synbiotorch.models.causal import CausalLMModel, build_causal_model
from synbiotorch.models.conv import ConvBackbone, SequenceBackbone, build_conv_backbone
from synbiotorch.models.heads import ClassificationHead, RegressionHead, build_head
from synbiotorch.models.mlm import MaskedLMModel, build_mlm_model
from synbiotorch.models.pooling import build_pooler
from synbiotorch.models.sequence_model import SequenceModel

__all__ = [
    "build_model",
    "build_architecture",
    "requires_fixed_length",
    "SequenceModel",
    "SequenceBackbone",
    "ConvBackbone",
    "MaskedLMModel",
    "CausalLMModel",
    "RegressionHead",
    "ClassificationHead",
    "build_head",
    "load_backbone",
]


def build_model(
    model_config: ModelConfig,
    task_config: TaskConfig,
    *,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
    max_length: int | None = None,
) -> nn.Module:
    """Build the model for the given task.

    - ``mlm`` → a MaskedLMModel (from-scratch needs ``vocab_size``/``pad_token_id``).
    - ``causal`` → a CausalLMModel decoder (from-scratch needs a decoder ``arch``,
      e.g. ``model_type: gpt2``).
    - ``frozen`` → a SequenceModel with a frozen backbone and a trainable head.
    - ``supervised`` → a SequenceModel fine-tuned end to end.

    ``max_length`` is the encoder's padded length, needed to size a ``flatten``
    pooler's head.
    """
    # Redundant with RunConfig validation on purpose: cli.py builds a TaskConfig
    # out of band, and tests and eval scripts call this directly.
    if task_config.kind != "supervised" and model_config.arch.family != "transformer":
        raise ConfigError(
            f"task.kind '{task_config.kind}' requires a transformer architecture; "
            f"model.arch.family is '{model_config.arch.family}'"
        )

    if task_config.kind == "mlm":
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required to build an MLM model")
        return build_mlm_model(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)

    if task_config.kind == "causal":
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required to build a causal LM model")
        return build_causal_model(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)

    backbone, channels = build_architecture(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)
    pooler = build_pooler(_pooling_kind(model_config), length=_pooled_length(model_config, backbone, max_length))
    head = build_head(task_config, pooler.output_dim(channels), model_config.dropout)
    return SequenceModel(backbone, head, pooler=pooler, freeze_backbone=task_config.kind == "frozen")


def build_architecture(
    model_config: ModelConfig,
    *,
    vocab_size: int | None = None,
    pad_token_id: int | None = None,
) -> tuple[nn.Module, int]:
    """Build the sequence backbone named by ``model.arch.family``.

    Returns ``(backbone, channels)``, where ``channels`` is the per-position
    feature width the pooler will reduce. The graph family is absent here
    because it consumes a different batch shape; ``build_graph_model`` serves
    that modality alongside ``GraphBatchAdapter``.
    """
    family = model_config.arch.family
    if family == "conv":
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required for a conv model")
        assert isinstance(model_config.arch, ConvArch)  # guaranteed by the family tag
        return build_conv_backbone(
            model_config.arch,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            dropout=model_config.dropout,
        )
    if family != "transformer":
        raise ConfigError(f"model.arch.family '{family}' cannot be built as a sequence backbone")
    if model_config.from_scratch:
        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id are required for a from-scratch model")
        return build_from_scratch_encoder(model_config, vocab_size=vocab_size, pad_token_id=pad_token_id)
    return load_backbone(model_config.backbone, model_config)


def _pooling_kind(model_config: ModelConfig) -> str:
    """Pooling for a sequence model. The graph family pools inside its own model."""
    arch = model_config.arch
    if isinstance(arch, GraphArch):
        raise ConfigError("the graph family does not use a sequence pooler")
    return arch.pooling


def _pooled_length(model_config: ModelConfig, backbone: nn.Module, max_length: int | None) -> int | None:
    """Sequence length reaching the pooler, for the poolers that need it."""
    if _pooling_kind(model_config) != "flatten" or max_length is None:
        return None
    return backbone.out_length(max_length) if isinstance(backbone, SequenceBackbone) else max_length


def requires_fixed_length(model_config: ModelConfig) -> bool:
    """Whether this model needs every batch padded to one length.

    ``flatten`` pooling yields a fixed-width vector, so a batch padded to its
    own longest member would give a record different features depending on
    which other records it travelled with.
    """
    return not isinstance(model_config.arch, GraphArch) and model_config.arch.pooling == "flatten"
