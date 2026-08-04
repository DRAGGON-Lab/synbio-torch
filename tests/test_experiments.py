from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import synbiotorch.cli as cli
from synbiotorch.experiments import (
    Experiment,
    ExperimentError,
    Qualification,
    Run,
    Variable,
)

ROOT = Path(__file__).parents[1]


def _titans_run(output_dir: Path) -> dict[str, Any]:
    return {
        "seed": 42,
        "output_dir": str(output_dir),
        "corpus": {
            "source": "synthetic",
            "n": 8,
            "synthetic_seed": 7,
            "cache_dir": str(output_dir / "cache"),
            "shard_size": 4,
        },
        "tokenizer": {"kind": "nonoverlap_kmer", "k": 1},
        "model": {
            "kind": "titans",
            "hidden_size": 8,
            "titans": {
                "num_layers": 1,
                "num_attention_heads": 2,
                "persistent_tokens": 2,
                "segment_length": 32,
                "memory_expansion_factor": 2,
            },
        },
        "task": {"kind": "causal"},
        "stream": {
            "enabled": True,
            "gradient_horizon": 1,
            "memory_mode": "adaptive",
            "shuffle_streams": False,
        },
        "splits": {"strategy": "random", "ratios": [0.75, 0.25, 0.0]},
        "train": {
            "batch_size": 2,
            "max_steps": 1,
            "eval_every_n_steps": 1,
            "amp": False,
            "precision": "fp32",
            "num_workers": 0,
        },
        "wandb": {"enabled": False},
    }


def _experiment(output_dir: Path) -> Experiment:
    return Experiment.model_validate(
        {
            "name": "titans-memory-inquiry",
            "run": _titans_run(output_dir),
            "variables": [
                {
                    "name": "memory_condition",
                    "path": "stream.memory_mode",
                    "values": ["adaptive", "frozen", "none"],
                },
                {"name": "seed", "path": "seed", "values": [42, 43]},
            ],
            "contrasts": [
                {
                    "name": "adaptive_vs_frozen",
                    "treatment": {"memory_condition": "adaptive"},
                    "control": {"memory_condition": "frozen"},
                    "metric": "bits_per_base",
                    "direction": "lower",
                }
            ],
            "invariants": ["stream_ownership", "higher_order_autograd"],
        }
    )


def test_experiment_resolves_directly_executable_run_configs(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "runs")
    run_set = experiment.resolve()
    repeated = experiment.resolve()

    assert len(run_set.runs) == 6
    assert [run.run_id for run in run_set.runs] == [run.run_id for run in repeated.runs]
    assert len({run.config.output_dir for run in run_set.runs}) == 6
    for run in run_set.runs:
        assert run.config.seed == run.values["seed"]
        assert run.config.stream.memory_mode == run.values["memory_condition"]
        assert Path(run.config.output_dir).name == run.run_id


def test_contrasts_automatically_match_every_other_variable(tmp_path: Path) -> None:
    run_set = _experiment(tmp_path / "runs").resolve()
    contrast = run_set.contrasts[0]
    runs = {run.run_id: run for run in run_set.runs}

    assert contrast.matched == ("seed",)
    assert len(contrast.pairs) == 2
    for pair in contrast.pairs:
        treatment = runs[pair.treatment]
        control = runs[pair.control]
        assert treatment.values["memory_condition"] == "adaptive"
        assert control.values["memory_condition"] == "frozen"
        assert treatment.values["seed"] == control.values["seed"]


def test_experiment_identity_ignores_labels_and_output_location(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "first")
    renamed = experiment.model_copy(
        update={
            "name": "presentation-label",
            "description": "non-semantic prose",
            "run": experiment.run.model_copy(update={"output_dir": str(tmp_path / "second")}),
        }
    )
    assert experiment.experiment_id == renamed.experiment_id
    reordered = experiment.model_copy(
        update={
            "variables": tuple(reversed(experiment.variables)),
            "invariants": tuple(reversed(experiment.invariants)),
        }
    )
    assert experiment.experiment_id == reordered.experiment_id
    different_template_defaults = experiment.model_copy(
        update={
            "run": experiment.run.model_copy(
                update={
                    "seed": 43,
                    "stream": experiment.run.stream.model_copy(update={"memory_mode": "frozen"}),
                }
            )
        }
    )
    assert experiment.experiment_id == different_template_defaults.experiment_id

    changed = experiment.model_copy(
        update={
            "variables": (
                experiment.variables[0].model_copy(update={"values": ("adaptive", "frozen")}),
                experiment.variables[1],
            )
        }
    )
    assert experiment.experiment_id != changed.experiment_id


def test_variable_path_and_resolved_run_are_validated(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "runs")
    unknown = experiment.model_copy(
        update={
            "variables": (Variable(name="unknown", path="train.not_a_field", values=(1, 2)),),
            "contrasts": (),
        }
    )
    with pytest.raises(ExperimentError, match="not a RunConfig field"):
        unknown.resolve()

    unsafe_amp = experiment.model_copy(
        update={
            "variables": (Variable(name="amp", path="train.amp", values=(False, True)),),
            "contrasts": (),
            "reference": {"amp": False},
            "qualifications": (
                Qualification(
                    when={"amp": True},
                    invariants=experiment.invariants,
                    evidence=("artifacts/amp-qualification.json",),
                ),
            ),
        }
    )
    with pytest.raises(ExperimentError, match="valid RunConfig"):
        unsafe_amp.resolve()


def test_execution_variables_fail_closed_without_complete_qualification(tmp_path: Path) -> None:
    base = {
        "corpus": {"source": "synthetic", "n": 8, "cache_dir": str(tmp_path / "cache")},
        "train": {"amp": False, "precision": "fp32"},
    }
    common = {
        "name": "precision-equivalence",
        "run": base,
        "variables": [
            {
                "name": "precision",
                "path": "train.precision",
                "values": ["fp32", "bf16"],
            }
        ],
        "reference": {"precision": "fp32"},
        "invariants": ["loss_equivalence", "metric_accounting"],
    }
    experiment = Experiment.model_validate(common)
    with pytest.raises(ExperimentError, match="exactly one qualification"):
        experiment.resolve()

    incomplete = Qualification(
        when={"precision": "bf16"},
        invariants=("loss_equivalence",),
        evidence=("artifacts/bf16.json",),
    )
    with pytest.raises(ExperimentError, match="metric_accounting"):
        experiment.model_copy(update={"qualifications": (incomplete,)}).resolve()

    complete = incomplete.model_copy(update={"invariants": experiment.invariants})
    run_set = experiment.model_copy(update={"qualifications": (complete,)}).resolve()
    assert len(run_set.runs) == 2
    assert run_set.runs[0].qualification is None
    assert run_set.runs[1].qualification == complete


def test_experiment_validation_rejects_inconsistent_assignments(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "runs")
    with pytest.raises(ExperimentError, match="unknown value"):
        Experiment.model_validate(
            {
                **experiment.model_dump(),
                "contrasts": [
                    {
                        "name": "invalid",
                        "treatment": {"memory_condition": "invented"},
                        "control": {"memory_condition": "frozen"},
                        "metric": "bits_per_base",
                        "direction": "lower",
                    }
                ],
            }
        )

    with pytest.raises(ValidationError):
        experiment.name = "mutated"  # type: ignore[misc]


def test_run_calls_training_and_records_provenance(tmp_path: Path) -> None:
    run = _experiment(tmp_path / "runs").resolve().runs[0]
    received: list[object] = []

    def fake_runner(config: object, *, resume_from: object = None) -> dict[str, float]:
        received.append((config, resume_from))
        return {"val_bits_per_base": 1.25}

    metrics = run.execute(runner=fake_runner)

    assert metrics == {"val_bits_per_base": 1.25}
    assert received == [(run.config, None)]
    manifest = json.loads((Path(run.config.output_dir) / "run.json").read_text())
    result = json.loads((Path(run.config.output_dir) / "result.json").read_text())
    assert manifest["run_id"] == run.run_id
    assert result["metrics"] == metrics


def test_resolved_run_executes_through_training_pipeline(tmp_path: Path) -> None:
    run = _experiment(tmp_path / "runs").resolve().runs[0]
    metrics = run.execute()

    assert "val_bits_per_base" in metrics
    assert (Path(run.config.output_dir) / "config.resolved.yaml").exists()
    assert (Path(run.config.output_dir) / "result.json").exists()


def test_example_and_cli_show_and_select_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = ROOT / "examples" / "experiments" / "titans_methodological_inquiry.yaml"
    experiment = Experiment.from_yaml(source)
    run_set = experiment.resolve()
    output = tmp_path / "runs.json"

    assert len(run_set.runs) == 6
    assert cli.main(["experiment", "show", str(source), "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["runs"]) == 6

    executed: list[str] = []

    def fake_execute(self: Run, **_: object) -> dict[str, float]:
        executed.append(self.run_id)
        return {"val_loss": 1.0}

    monkeypatch.setattr(Run, "execute", fake_execute)
    selected = run_set.runs[2].run_id
    assert cli.main(["experiment", "run", str(source), "--run-id", selected]) == 0
    assert executed == [selected]
