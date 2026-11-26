"""Target transforms on the supervised task: round-trip fidelity and validation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from synbiotorch.config import TaskConfig
from synbiotorch.exceptions import ConfigError
from synbiotorch.tasks import SupervisedTask, build_task

LABELS = torch.tensor([0.0, 1.0, 2.5, 17.0, 132.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_transform": "none"},
        {"target_transform": "log1p"},
        {"target_transform": "standardize", "target_mean": 2.3178, "target_std": 6.5111},
    ],
)
def test_inverse_transform_round_trips(kwargs: dict) -> None:
    task = SupervisedTask(objective="regression", **kwargs)
    forward = task._forward_transform(LABELS).numpy()
    assert np.allclose(task.inverse_transform(forward), LABELS.numpy(), rtol=1e-5, atol=1e-5)


def test_standardize_centers_and_scales() -> None:
    task = SupervisedTask(objective="regression", target_transform="standardize", target_mean=2.0, target_std=4.0)
    assert np.allclose(task._forward_transform(torch.tensor([2.0, 6.0, -2.0])).numpy(), [0.0, 1.0, -1.0])


def test_transforms_do_not_apply_to_classification() -> None:
    task = SupervisedTask(objective="classification", num_classes=2, target_transform="log1p")
    labels = torch.tensor([0, 1, 1])
    assert torch.equal(task._forward_transform(labels), labels)


def test_build_task_passes_standardize_statistics() -> None:
    task = build_task(
        TaskConfig(
            kind="supervised",
            objective="regression",
            target_transform="standardize",
            target_mean=2.5,
            target_std=0.5,
        )
    )
    assert isinstance(task, SupervisedTask)
    assert (task.target_mean, task.target_std) == (2.5, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_mean": 1.0},  # std missing
        {"target_std": 1.0},  # mean missing
        {},  # both missing
        {"target_mean": 1.0, "target_std": 0.0},  # degenerate scale
    ],
)
def test_standardize_requires_usable_statistics(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        TaskConfig(kind="supervised", objective="regression", target_transform="standardize", **kwargs)
