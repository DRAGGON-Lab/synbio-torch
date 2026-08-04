"""Parameter-sweep experiments that resolve directly to executable runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from itertools import product
from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from synbiotorch.config import RunConfig
from synbiotorch.exceptions import ConfigError

Scalar: TypeAlias = str | int | float | bool | None
Runner: TypeAlias = Callable[..., dict[str, float]]


class ExperimentError(ConfigError):
    """An experiment cannot resolve its declared runs."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _semantic_id(prefix: str, value: object) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()}"


def _config_semantics(config: RunConfig) -> dict[str, object]:
    return config.model_dump(mode="json", exclude={"output_dir"})


def _set_path(document: dict[str, object], path: str, value: Scalar) -> None:
    current: MutableMapping[str, object] = document
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            raise ExperimentError(f"variable path {path!r} is not a RunConfig field")
        current = child
    if parts[-1] not in current:
        raise ExperimentError(f"variable path {path!r} is not a RunConfig field")
    current[parts[-1]] = value


class Variable(_Model):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    path: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)*$")
    values: tuple[Scalar, ...]

    @model_validator(mode="after")
    def _check_values(self) -> "Variable":
        if self.path == "output_dir":
            raise ExperimentError("output_dir is assigned automatically")
        identities = [_semantic_id("value", value) for value in self.values]
        if len(self.values) < 2:
            raise ExperimentError(f"variable {self.name!r} requires at least two values")
        if len(set(identities)) != len(identities):
            raise ExperimentError(f"variable {self.name!r} values must be unique")
        return self


class Contrast(_Model):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    treatment: dict[str, Scalar]
    control: dict[str, Scalar]
    metric: str = Field(min_length=1)
    direction: Literal["lower", "higher", "two_sided"]


class Qualification(_Model):
    when: dict[str, Scalar]
    invariants: tuple[str, ...]
    evidence: tuple[str, ...]

    @model_validator(mode="after")
    def _check_evidence(self) -> "Qualification":
        if not self.when:
            raise ExperimentError("qualification requires a variable condition")
        if not self.evidence or any(not path for path in self.evidence):
            raise ExperimentError("qualification requires evidence paths")
        if any(not invariant for invariant in self.invariants) or len(set(self.invariants)) != len(self.invariants):
            raise ExperimentError("qualification invariants must be named and unique")
        return self


class RunPair(_Model):
    treatment: str
    control: str


class ResolvedContrast(_Model):
    name: str
    metric: str
    direction: Literal["lower", "higher", "two_sided"]
    matched: tuple[str, ...]
    pairs: tuple[RunPair, ...]


class Run(_Model):
    run_id: str
    experiment_id: str
    values: dict[str, Scalar]
    config: RunConfig
    invariants: tuple[str, ...] = ()
    qualification: Qualification | None = None

    def execute(self, *, resume_from: str | Path | None = None, runner: Runner | None = None) -> dict[str, float]:
        """Execute this run through the ordinary training pipeline."""

        if runner is None:
            from synbiotorch.pipeline import run_training

            runner = run_training
        output = Path(self.config.output_dir)
        is_main = os.environ.get("RANK", "0") == "0"
        if is_main:
            output.mkdir(parents=True, exist_ok=True)
            (output / "run.json").write_text(f"{self.model_dump_json(indent=2)}\n")
        metrics = runner(self.config, resume_from=resume_from)
        if is_main:
            result = {"run_id": self.run_id, "experiment_id": self.experiment_id, "metrics": metrics}
            (output / "result.json").write_text(f"{json.dumps(result, indent=2, sort_keys=True)}\n")
        return metrics


class RunSet(_Model):
    experiment_id: str
    runs: tuple[Run, ...]
    contrasts: tuple[ResolvedContrast, ...]

    def select(self, run_ids: Iterable[str] = ()) -> tuple[Run, ...]:
        requested = set(run_ids)
        if not requested:
            return self.runs
        selected = tuple(run for run in self.runs if run.run_id in requested)
        missing = requested - {run.run_id for run in selected}
        if missing:
            raise ExperimentError(f"unknown run ids: {sorted(missing)}")
        return selected

    def execute(self, run_ids: Iterable[str] = (), *, runner: Runner | None = None) -> dict[str, dict[str, float]]:
        return {run.run_id: run.execute(runner=runner) for run in self.select(run_ids)}


class Experiment(_Model):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    description: str | None = None
    run: RunConfig
    variables: tuple[Variable, ...]
    contrasts: tuple[Contrast, ...] = ()
    reference: dict[str, Scalar] = Field(default_factory=dict)
    invariants: tuple[str, ...] = ()
    qualifications: tuple[Qualification, ...] = ()

    @model_validator(mode="after")
    def _check_experiment(self) -> "Experiment":
        variables = {variable.name: variable for variable in self.variables}
        if not variables:
            raise ExperimentError("an experiment requires at least one variable")
        if len(variables) != len(self.variables):
            raise ExperimentError("variable names must be unique")
        if len({variable.path for variable in self.variables}) != len(self.variables):
            raise ExperimentError("variable paths must be unique")
        if len({contrast.name for contrast in self.contrasts}) != len(self.contrasts):
            raise ExperimentError("contrast names must be unique")
        if any(not invariant for invariant in self.invariants) or len(set(self.invariants)) != len(self.invariants):
            raise ExperimentError("invariants must be named and unique")
        self._check_values("reference", self.reference, variables)
        for contrast in self.contrasts:
            if set(contrast.treatment) != set(contrast.control):
                raise ExperimentError(f"contrast {contrast.name!r} arms must assign the same variables")
            if contrast.treatment == contrast.control:
                raise ExperimentError(f"contrast {contrast.name!r} arms must differ")
            self._check_values(f"contrast {contrast.name!r}", contrast.treatment, variables)
            self._check_values(f"contrast {contrast.name!r}", contrast.control, variables)
        for qualification in self.qualifications:
            self._check_values("qualification", qualification.when, variables)
        return self

    @staticmethod
    def _check_values(label: str, values: dict[str, Scalar], variables: dict[str, Variable]) -> None:
        unknown = set(values) - set(variables)
        if unknown:
            raise ExperimentError(f"{label} references unknown variables: {sorted(unknown)}")
        for name, value in values.items():
            allowed = {_semantic_id("value", candidate) for candidate in variables[name].values}
            if _semantic_id("value", value) not in allowed:
                raise ExperimentError(f"{label} uses an unknown value for variable {name!r}")

    @property
    def experiment_id(self) -> str:
        run = _config_semantics(self.run)
        variables = []
        for variable in sorted(self.variables, key=lambda item: item.name):
            _set_path(run, variable.path, None)
            item = variable.model_dump(mode="json")
            item["values"] = sorted(item["values"], key=lambda value: _semantic_id("value", value))
            variables.append(item)
        semantic = {
            "run": run,
            "variables": variables,
            "contrasts": sorted(self.contrasts, key=lambda item: item.name),
            "reference": self.reference,
            "invariants": sorted(self.invariants),
        }
        return _semantic_id("experiment", semantic)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Experiment":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})

    def resolve(self) -> RunSet:
        variables = {variable.name: variable for variable in self.variables}
        combinations = product(*(variable.values for variable in self.variables))
        runs = tuple(
            self._resolve_run(
                {variable.name: value for variable, value in zip(self.variables, combination)},
                variables,
            )
            for combination in combinations
        )
        contrasts = tuple(self._resolve_contrast(contrast, runs) for contrast in self.contrasts)
        return RunSet(experiment_id=self.experiment_id, runs=runs, contrasts=contrasts)

    def _resolve_run(self, values: dict[str, Scalar], variables: dict[str, Variable]) -> Run:
        document: dict[str, object] = self.run.model_dump(mode="json")
        for name, value in values.items():
            _set_path(document, variables[name].path, value)
        try:
            config = RunConfig.model_validate(document)
        except (ValidationError, ConfigError) as error:
            raise ExperimentError(f"values {values} do not produce a valid RunConfig: {error}") from error
        qualification = self._qualification(values)
        run_id = _semantic_id(
            "run",
            {"experiment_id": self.experiment_id, "values": values, "config": _config_semantics(config)},
        )
        config = config.model_copy(update={"output_dir": str(Path(self.run.output_dir) / run_id)})
        return Run(
            run_id=run_id,
            experiment_id=self.experiment_id,
            values=values,
            config=config,
            invariants=self.invariants,
            qualification=qualification,
        )

    def _qualification(self, values: dict[str, Scalar]) -> Qualification | None:
        if all(values[name] == value for name, value in self.reference.items()):
            return None
        matches = [
            qualification
            for qualification in self.qualifications
            if all(values.get(name) == value for name, value in qualification.when.items())
        ]
        if len(matches) != 1:
            raise ExperimentError(f"non-reference values require exactly one qualification: {values}")
        missing = set(self.invariants) - set(matches[0].invariants)
        if missing:
            raise ExperimentError(f"qualification for {values} lacks invariants: {sorted(missing)}")
        return matches[0]

    def _resolve_contrast(self, contrast: Contrast, runs: tuple[Run, ...]) -> ResolvedContrast:
        matched = tuple(variable.name for variable in self.variables if variable.name not in contrast.treatment)
        treatment: dict[tuple[Scalar, ...], list[Run]] = defaultdict(list)
        control: dict[tuple[Scalar, ...], list[Run]] = defaultdict(list)
        for run in runs:
            key = tuple(run.values[name] for name in matched)
            if all(run.values.get(name) == value for name, value in contrast.treatment.items()):
                treatment[key].append(run)
            if all(run.values.get(name) == value for name, value in contrast.control.items()):
                control[key].append(run)
        if not treatment or not control or set(treatment) != set(control):
            raise ExperimentError(f"contrast {contrast.name!r} has unmatched arms")
        pairs = []
        for key in sorted(treatment, key=lambda value: json.dumps(value, sort_keys=True)):
            if len(treatment[key]) != 1 or len(control[key]) != 1:
                raise ExperimentError(f"contrast {contrast.name!r} is not one-to-one")
            pairs.append(RunPair(treatment=treatment[key][0].run_id, control=control[key][0].run_id))
        return ResolvedContrast(
            name=contrast.name,
            metric=contrast.metric,
            direction=contrast.direction,
            matched=matched,
            pairs=tuple(pairs),
        )
