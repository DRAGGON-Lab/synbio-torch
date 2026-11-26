"""The conv architecture family: shapes, mask propagation, and pooling."""

from __future__ import annotations

import pytest
import torch

from synbiotorch.config import ArchConfig, ConvArch, ModelConfig, TaskConfig
from synbiotorch.exceptions import ConfigError
from synbiotorch.models import build_model, requires_fixed_length
from synbiotorch.models.conv import ConvBackbone
from synbiotorch.models.pooling import ClsPooler, FlattenPooler, MaxPooler, MeanPooler, build_pooler
from synbiotorch.tokenize.char import CharTokenizer

# The architecture Urtecho et al. selected by random hyperparameter search.
URTECHO_REGRESSION = ConvArch(
    family="conv",
    channels=(19, 39, 54),
    kernel_sizes=(16, 17, 18),
    pool_width=16,
    embed_dim=4,
    pooling="flatten",
)
LENGTH = 152  # 150 bp + <cls> + <sep>


def _tokenizer() -> CharTokenizer:
    return CharTokenizer(alphabet="dna", max_length=LENGTH)


def _backbone(arch: ConvArch = URTECHO_REGRESSION) -> ConvBackbone:
    tok = _tokenizer()
    return ConvBackbone(arch, vocab_size=tok.vocab_size, pad_token_id=tok.pad_token_id, dropout=0.0)


def test_out_length_is_pure_arithmetic() -> None:
    """The head must be sizeable before any tensor exists."""
    backbone = _backbone()
    assert backbone.out_length(LENGTH) == 6
    hidden, _ = backbone(torch.zeros(2, LENGTH, dtype=torch.long), torch.ones(2, LENGTH, dtype=torch.long))
    assert hidden.shape == (2, backbone.out_length(LENGTH), backbone.out_channels)


def test_urtecho_parameter_count() -> None:
    """Pin the reproduction: 52,158 conv+dense parameters plus an 84-parameter embedding."""
    model = build_model(
        ModelConfig(dropout=0.134, arch=URTECHO_REGRESSION),
        TaskConfig(kind="supervised", objective="regression"),
        vocab_size=_tokenizer().vocab_size,
        pad_token_id=_tokenizer().pad_token_id,
        max_length=LENGTH,
    )
    assert sum(p.numel() for p in model.parameters()) == 52_242


def test_checkpoint_keys_are_unchanged() -> None:
    """Existing checkpoints load strictly, so the top-level names must not drift."""
    model = build_model(
        ModelConfig(from_scratch=True, hidden_size=32, arch=ArchConfig(num_hidden_layers=1, num_attention_heads=2)),
        TaskConfig(kind="supervised", objective="regression"),
        vocab_size=16,
        pad_token_id=0,
    )
    assert {key.split(".")[0] for key in model.state_dict()} == {"backbone", "head"}


def test_mask_erodes_through_convolutions_and_dilates_through_pooling() -> None:
    backbone = _backbone()
    mask = torch.ones(1, LENGTH, dtype=torch.long)
    mask[0, 100:] = 0  # 100 valid positions
    _, out_mask = backbone(torch.zeros(1, LENGTH, dtype=torch.long), mask)
    # Each valid convolution shortens the valid run by (kernel - 1); the
    # max-pool then keeps any bin containing a valid position.
    valid = 100
    for width in URTECHO_REGRESSION.kernel_sizes:
        valid -= width - 1
    assert int(out_mask.sum()) == -(-valid // URTECHO_REGRESSION.pool_width)  # ceil


def test_padding_does_not_change_a_records_features() -> None:
    """The property adaptive pooling would violate.

    A record padded alongside longer neighbours must encode identically to the
    same record on its own, or metrics would depend on batch composition.
    """
    torch.manual_seed(0)
    backbone = _backbone().eval()
    pooler = FlattenPooler(backbone.out_length(LENGTH))
    ids = torch.randint(1, 10, (1, LENGTH))
    mask = torch.ones(1, LENGTH, dtype=torch.long)
    mask[0, 90:] = 0

    alone = pooler(*backbone(ids, mask))
    batched_ids = torch.cat([ids, torch.randint(1, 10, (3, LENGTH))])
    batched_mask = torch.cat([mask, torch.ones(3, LENGTH, dtype=torch.long)])
    together = pooler(*backbone(batched_ids, batched_mask))[:1]
    assert torch.allclose(alone, together, atol=1e-6)


def test_mean_pooler_matches_the_historical_masked_mean() -> None:
    """MeanPooler is the default for every pre-existing run; it must not drift."""
    torch.manual_seed(0)
    hidden = torch.randn(4, 7, 5)
    mask = torch.ones(4, 7, dtype=torch.long)
    mask[1, 3:] = 0
    expected = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.unsqueeze(-1).sum(dim=1).clamp(min=1e-9)
    assert torch.allclose(MeanPooler()(hidden, mask), expected, atol=1e-6)


def test_max_pooler_is_finite_for_a_fully_masked_row() -> None:
    """A dtype-minimum fill must never reach the head."""
    hidden = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.long)
    mask[0, 0] = 1
    pooled = MaxPooler()(hidden, mask)
    assert torch.isfinite(pooled).all()
    assert torch.equal(pooled[1], torch.zeros(3))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("mean", 8), ("max", 8), ("cls", 8), ("flatten", 24)],
)
def test_pooler_output_dim(kind: str, expected: int) -> None:
    assert build_pooler(kind, length=3).output_dim(8) == expected


def test_flatten_pooler_needs_a_length() -> None:
    with pytest.raises(ConfigError):
        build_pooler("flatten")


def test_build_pooler_kinds() -> None:
    assert isinstance(build_pooler("mean"), MeanPooler)
    assert isinstance(build_pooler("max"), MaxPooler)
    assert isinstance(build_pooler("cls"), ClsPooler)


def test_resolved_config_round_trips() -> None:
    """``config.resolved.yaml`` is dumped with every field and must reload.

    Family-specific rejections therefore cannot key off a field merely being
    present, or a conv run could never be re-scored from its own artifact.
    """
    import yaml

    from synbiotorch.config import CorpusConfig, EncoderConfig, RunConfig, SplitConfig, TokenizerConfig

    config = RunConfig(
        corpus=CorpusConfig(source="table", path="x.csv", sequence_column="sequence", label_column="label"),
        tokenizer=TokenizerConfig(kind="char", alphabet="dna", max_length=LENGTH),
        encoder=EncoderConfig(kind="sequence"),
        model=ModelConfig(dropout=0.134, arch=URTECHO_REGRESSION),
        task=TaskConfig(kind="supervised", objective="regression"),
        splits=SplitConfig(strategy="column", column="split"),
    )
    reloaded = RunConfig.model_validate(yaml.safe_load(config.to_yaml()))
    assert reloaded.model.arch.family == "conv"
    # Normalized, so the artifact records what was actually built.
    assert reloaded.model.from_scratch is True
    assert reloaded.to_yaml() == config.to_yaml()


def test_explicit_from_scratch_false_is_rejected_for_conv() -> None:
    with pytest.raises(ConfigError):
        ModelConfig(from_scratch=False, arch=URTECHO_REGRESSION)


def test_meaningful_transformer_only_fields_are_rejected_for_conv() -> None:
    with pytest.raises(ConfigError):
        ModelConfig(hidden_size=256, arch=URTECHO_REGRESSION)
    with pytest.raises(ConfigError):
        ModelConfig(backbone="zhihan1996/DNABERT-2-117M-other", arch=URTECHO_REGRESSION)


def test_mistagged_arch_fields_are_rejected() -> None:
    """extra='forbid' stops a stale transformer field silently training a default CNN."""
    with pytest.raises(Exception):
        ModelConfig(arch={"family": "conv", "model_type": "bert", "num_hidden_layers": 4})


def test_requires_fixed_length_only_for_flatten() -> None:
    assert requires_fixed_length(ModelConfig(arch=URTECHO_REGRESSION))
    assert not requires_fixed_length(ModelConfig(arch=URTECHO_REGRESSION.model_copy(update={"pooling": "max"})))
    assert not requires_fixed_length(ModelConfig(from_scratch=True, arch=ArchConfig()))
