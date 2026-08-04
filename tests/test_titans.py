"""Paper-deep TITANS MAC mathematics, state isolation, and engine integration."""

from __future__ import annotations

import math
from collections import OrderedDict

import pytest
import torch
from torch.utils.data import DataLoader

from synbiotorch.config import (
    CorpusConfig,
    ModelConfig,
    RunConfig,
    SplitConfig,
    StreamConfig,
    TaskConfig,
    TitansConfig,
    TokenizerConfig,
    TrainConfig,
)
from synbiotorch.datasets.stream_windows import OrderedStreamWindowDataset
from synbiotorch.engine.stateful import StatefulCausalBatchAdapter
from synbiotorch.engine.trainer import Callback, Trainer
from synbiotorch.exceptions import ConfigError
from synbiotorch.generate import generate_titans
from synbiotorch.models.titans import (
    MACBlock,
    NeuralMemory,
    TitansCausalLM,
    mac_attention_mask,
)
from synbiotorch.pipeline import run_training
from synbiotorch.reproducibility import set_seed
from synbiotorch.tasks.causal import CausalLMTask
from synbiotorch.tokenize.kmer import NonOverlappingKmerTokenizer
from synbiotorch.types import Alphabet, Design, Sequence

CPU = torch.device("cpu")


def _architecture(*, layers: int = 1, heads: int = 2) -> TitansConfig:
    return TitansConfig(
        num_layers=layers,
        num_attention_heads=heads,
        persistent_tokens=2,
        memory_expansion_factor=2,
    )


def _model(*, layers: int = 1, vocab_size: int = 32) -> TitansCausalLM:
    return TitansCausalLM(
        _architecture(layers=layers),
        vocab_size=vocab_size,
        pad_token_id=0,
        d_model=8,
    )


def _design(name: str, sequence: str) -> Design:
    return Design(
        iri=f"https://example.org/{name}",
        record_class="http://sbols.org/v3#Sequence",
        sequence=Sequence(elements=sequence, alphabet=Alphabet.DNA),
    )


def test_mac_mask_blocks_every_future_retrieval_and_token():
    mask = mac_attention_mask(2, 32)
    retrieval_start = 2
    sequence_start = 34
    for position in range(32):
        assert not mask[sequence_start + position, :2].any()
        assert not mask[sequence_start + position, retrieval_start : retrieval_start + position + 1].any()
        assert not mask[sequence_start + position, sequence_start : sequence_start + position + 1].any()
        if position + 1 < 32:
            assert mask[sequence_start + position, retrieval_start + position + 1 : 34].all()
            assert mask[sequence_start + position, sequence_start + position + 1 :].all()


def test_mac_future_values_cannot_change_earlier_outputs():
    set_seed(1)
    block = MACBlock(
        8,
        num_heads=2,
        persistent_tokens=2,
        segment_length=32,
        memory_expansion_factor=2,
        projection_kernel_size=4,
        alpha_initial=0.001,
        eta_initial=0.9,
        theta_initial=0.001,
    ).double()
    retrieval = torch.randn(32, 8, dtype=torch.float64)
    segment = torch.randn(32, 8, dtype=torch.float64)
    baseline = block.integrate(retrieval, segment)
    retrieval[20:] += 100.0
    segment[20:] -= 100.0
    changed = block.integrate(retrieval, segment)
    torch.testing.assert_close(baseline[:20], changed[:20], rtol=0, atol=1e-10)


def test_memory_one_position_matches_the_declared_recurrence():
    set_seed(2)
    memory = NeuralMemory(4, expansion_factor=2, projection_kernel_size=4).double()
    state = memory.initial_state()
    inputs = torch.randn(32, 4, dtype=torch.float64)
    valid = torch.zeros(32, dtype=torch.bool)
    valid[0] = True
    _, query_history = memory.read_with_history(state, inputs)
    keys, values, _ = memory._project_writes(state, inputs)
    gates = memory.gates(inputs)
    gradient = memory._surprise_gradient(state.fast_weights, keys[0], values[0], create_graph=True)
    expected_surprise = OrderedDict(
        (
            name,
            gates.eta[name][0] * state.surprise[name] - gates.theta[name][0] * gradient[name],
        )
        for name in state.fast_weights
    )
    expected_weights = OrderedDict(
        (
            name,
            (1.0 - gates.alpha[name][0]) * state.fast_weights[name] + expected_surprise[name],
        )
        for name in state.fast_weights
    )
    updated = memory.update(
        state,
        inputs,
        valid_mask=valid,
        query_history=query_history,
        create_graph=True,
    )
    for name in expected_weights:
        torch.testing.assert_close(updated.surprise[name], expected_surprise[name])
        torch.testing.assert_close(updated.fast_weights[name], expected_weights[name])


def test_model_outer_gradient_reaches_memory_initialization_and_gate_heads():
    set_seed(3)
    model = _model()
    ids = torch.randint(2, model.vocab_size, (1, 32))
    first = model.forward_segment(ids)
    second = model.forward_segment(ids, first.states)
    second.logits.square().mean().backward()
    assert model.blocks[0].memory.memory.in_projection.weight.grad is not None
    assert model.blocks[0].memory.gates.heads["in_projection"].weight.grad is not None


def test_cpu_bfloat16_autocast_keeps_higher_order_memory_state_in_float32():
    set_seed(31)
    model = _model()
    ids = torch.randint(2, model.vocab_size, (1, 32))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = model.forward_segment(ids)
        second = model.forward_segment(ids, first.states)
        loss = second.logits.float().square().mean()
    loss.backward()

    memory_state = second.states[0].blocks[0]
    state_tensors = [
        *memory_state.fast_weights.values(),
        *memory_state.surprise.values(),
        memory_state.query_history,
        memory_state.write_history,
    ]
    assert all(tensor.dtype == torch.float32 for tensor in state_tensors)
    assert all(torch.isfinite(tensor).all() for tensor in state_tensors)
    memory = model.blocks[0].memory
    for parameter in (memory.memory.in_projection.weight, memory.gates.heads["in_projection"].weight):
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()


def test_no_memory_inference_is_a_fullgraph_compile_boundary():
    set_seed(32)
    model = _model().eval()
    ids = torch.randint(2, model.vocab_size, (1, 32))

    def no_memory(input_ids: torch.Tensor) -> torch.Tensor:
        return model.forward_segment(input_ids, memory_mode="none").logits

    expected = no_memory(ids)
    compiled = torch.compile(no_memory, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(ids), expected)


def test_stream_windows_preserve_cross_segment_labels_and_base_counts():
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    sequence = "ACGT" * 30
    dataset = OrderedStreamWindowDataset(
        [_design("a", sequence), _design("b", sequence[::-1])],
        tokenizer,
        batch_size=2,
        segment_length=32,
        gradient_horizon=2,
        shuffle_streams=False,
    )
    batch = next(iter(dataset))
    assert batch["input_ids"].shape == (2, 2, 32)
    assert batch["labels"][0, 0, -1] == batch["input_ids"][0, 1, 0]
    assert int(batch["represented_base_counts"].sum()) == 2 * 64
    assert batch["stream_ids"] == ("https://example.org/a", "https://example.org/b")


def test_bits_per_base_uses_additive_nll_and_represented_base_counts():
    adapter = StatefulCausalBatchAdapter()
    labels = torch.tensor([[[2, 3, -100]]])
    logits = torch.zeros((1, 1, 3, 4))
    logits[0, 0, 0, 2] = math.log(3.0)
    logits[0, 0, 1, 3] = math.log(7.0)
    batch = {"represented_base_counts": torch.tensor([[[6, 1, 0]]])}
    contributions = adapter.metric_contributions(logits, labels, batch)
    nll_sum, token_count = contributions["loss"]
    bit_sum, base_count = contributions["bits_per_base"]
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 4), labels.reshape(-1), ignore_index=-100, reduction="sum"
    )
    torch.testing.assert_close(nll_sum, expected)
    assert int(token_count) == 2
    assert int(base_count) == 7
    torch.testing.assert_close(bit_sum / base_count, expected / (7 * math.log(2.0)))


def test_bits_per_base_rejects_unrepresented_active_labels():
    adapter = StatefulCausalBatchAdapter()
    labels = torch.tensor([[[2]]])
    logits = torch.zeros((1, 1, 1, 4))
    batch = {"represented_base_counts": torch.zeros_like(labels)}
    with pytest.raises(ValueError, match="positive represented-base count"):
        adapter.metric_contributions(logits, labels, batch)


def test_stateful_adapter_isolates_streams_and_serializes_active_state():
    set_seed(4)
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    dataset = OrderedStreamWindowDataset(
        [_design("a", "ACGT" * 80), _design("b", "TGCA" * 80)],
        tokenizer,
        batch_size=2,
        segment_length=32,
        gradient_horizon=2,
        shuffle_streams=False,
    )
    batch = next(iter(dataset))
    adapter = StatefulCausalBatchAdapter()
    model = _model(vocab_size=tokenizer.vocab_size)
    logits = adapter.forward(model, batch)
    assert logits.shape == (2, 2, 32, tokenizer.vocab_size)
    assert set(adapter.stream_states) == set(batch["stream_ids"])
    first = adapter.stream_states["https://example.org/a"]
    second = adapter.stream_states["https://example.org/b"]
    assert first is not second
    restored = StatefulCausalBatchAdapter()
    restored.load_state_dict(adapter.state_dict())
    assert set(restored.stream_states) == set(adapter.stream_states)


class _StopAtStep(Callback):
    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(self, trainer: Trainer, step: int, logs: dict[str, float]) -> None:
        if step == self.step:
            trainer.should_stop = True


def _stateful_training_components(seed: int = 12):
    tokenizer = NonOverlappingKmerTokenizer(k=1)
    designs = [
        _design("a", "ACGT" * 100),
        _design("b", "TGCA" * 100),
        _design("c", "GGCC" * 100),
    ]
    dataset = OrderedStreamWindowDataset(
        designs,
        tokenizer,
        batch_size=2,
        segment_length=32,
        gradient_horizon=1,
        seed=seed,
        shuffle_streams=False,
    )
    return tokenizer, DataLoader(dataset, batch_size=None), StatefulCausalBatchAdapter()


def test_titans_checkpoint_resumes_model_cursor_and_fast_state_exactly(tmp_path):
    config = TrainConfig(
        batch_size=2,
        max_steps=2,
        lr=1e-3,
        amp=False,
        precision="fp32",
    )

    set_seed(11)
    tokenizer, loader, adapter = _stateful_training_components()
    full = Trainer(
        _model(vocab_size=tokenizer.vocab_size),
        CausalLMTask(),
        config,
        batch_adapter=adapter,
        device=CPU,
    )
    full.fit(loader)

    set_seed(11)
    tokenizer, loader, adapter = _stateful_training_components()
    partial = Trainer(
        _model(vocab_size=tokenizer.vocab_size),
        CausalLMTask(),
        config,
        callbacks=[_StopAtStep(1)],
        batch_adapter=adapter,
        device=CPU,
    )
    partial.fit(loader)
    checkpoint = tmp_path / "titans.pt"
    partial.save_checkpoint(checkpoint, epoch=partial.current_epoch, metrics={})

    set_seed(99)
    tokenizer, loader, adapter = _stateful_training_components()
    resumed = Trainer(
        _model(vocab_size=tokenizer.vocab_size),
        CausalLMTask(),
        config,
        batch_adapter=adapter,
        device=CPU,
    )
    resumed.fit(loader, resume_from=checkpoint)

    assert resumed.global_step == full.global_step == 2
    for name, expected in full._base_model.state_dict().items():
        torch.testing.assert_close(resumed._base_model.state_dict()[name], expected, rtol=0, atol=0)
    assert set(resumed.adapter.stream_states) == set(full.adapter.stream_states)
    for stream_id in full.adapter.stream_states:
        expected = full.adapter.stream_states[stream_id]
        actual = resumed.adapter.stream_states[stream_id]
        for expected_block, actual_block in zip(expected.blocks, actual.blocks):
            for name in expected_block.fast_weights:
                torch.testing.assert_close(
                    actual_block.fast_weights[name],
                    expected_block.fast_weights[name],
                    rtol=0,
                    atol=0,
                )


def test_titans_generation_is_seeded_and_segment_aware():
    set_seed(5)
    model = _model(vocab_size=16)
    prompt = [2, 3, 4]
    first = generate_titans(
        model,
        prompt,
        max_new_tokens=4,
        seed=9,
        forbidden_token_ids=(0, 1),
    )
    second = generate_titans(
        model,
        prompt,
        max_new_tokens=4,
        seed=9,
        forbidden_token_ids=(0, 1),
    )
    assert first == second and len(first) == 7


def test_titans_config_rejects_unsafe_initial_execution_modes():
    base = {
        "corpus": {"source": "synthetic"},
        "tokenizer": {"kind": "nonoverlap_kmer", "k": 6},
        "model": {"kind": "titans_mac", "hidden_size": 8, "titans": {"num_attention_heads": 2}},
        "task": {"kind": "causal"},
        "stream": {"enabled": True},
        "train": {"max_steps": 1, "amp": False, "precision": "fp32"},
    }
    assert RunConfig.model_validate(base).model.kind == "titans_mac"
    unsafe = [
        {**base, "streaming": True},
        {**base, "packing": {"enabled": True}},
        {**base, "train": {"max_steps": 1, "amp": False, "num_workers": 1}},
        {**base, "train": {"max_steps": 1, "amp": False, "distributed": {"strategy": "ddp"}}},
        {**base, "train": {"max_steps": 1, "amp": True}},
        {**base, "train": {"max_steps": 1, "amp": False, "compile": True}},
        {**base, "train": {"max_steps": 1, "amp": False, "gradient_checkpointing": True}},
    ]
    for candidate in unsafe:
        with pytest.raises(ConfigError):
            RunConfig.model_validate(candidate)


def test_titans_pipeline_runs_through_existing_trainer(tmp_path):
    config = RunConfig(
        output_dir=str(tmp_path / "run"),
        corpus=CorpusConfig(
            source="synthetic",
            n=16,
            shard_size=8,
            cache_dir=str(tmp_path / "cache"),
        ),
        tokenizer=TokenizerConfig(kind="nonoverlap_kmer", k=1),
        model=ModelConfig(
            kind="titans_mac",
            hidden_size=8,
            titans=_architecture(),
        ),
        task=TaskConfig(kind="causal"),
        stream=StreamConfig(enabled=True, gradient_horizon=1),
        splits=SplitConfig(strategy="random", ratios=(0.75, 0.25, 0.0)),
        train=TrainConfig(
            batch_size=2,
            max_steps=1,
            eval_every_n_steps=1,
            amp=False,
            precision="fp32",
        ),
    )
    metrics = run_training(config)
    assert "val_loss" in metrics
    assert metrics["train_bits_per_base"] == pytest.approx(metrics["train_loss"] / math.log(2.0))
    assert metrics["val_bits_per_base"] == pytest.approx(metrics["val_loss"] / math.log(2.0))
    assert (tmp_path / "run" / "best.pt").exists()
