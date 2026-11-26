"""Supervised regression / classification objective."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SupervisedTask:
    """Regression (MSE) or classification (cross-entropy) with optional target transform.

    ``target_transform='log1p'`` trains in log space (common for expression and
    fitness targets, as in the original SeqTrainer DNABERT path) and reports
    metrics back in the original space via ``expm1``.

    ``target_transform='standardize'`` centers and scales the target by
    ``target_mean``/``target_std``, which are the train-partition statistics.
    Being affine it leaves the objective and the reported metrics unchanged, and
    exists for targets whose raw scale makes the squared-error gradient
    ill-conditioned: with a heavy-tailed target, the informative residuals are
    large enough that gradient clipping discards them and the model settles on
    predicting the mean.
    """

    def __init__(
        self,
        objective: str = "regression",
        num_classes: int | None = None,
        target_transform: str = "none",
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> None:
        self.objective = objective
        self.num_classes = num_classes
        self.target_transform = target_transform
        self.target_mean = target_mean
        self.target_std = target_std
        self.label_dtype = "long" if objective == "classification" else "float"

    def _transforms(self) -> bool:
        return self.objective == "regression"

    def _forward_transform(self, labels: torch.Tensor) -> torch.Tensor:
        if not self._transforms():
            return labels
        if self.target_transform == "log1p":
            return torch.log1p(labels)
        if self.target_transform == "standardize":
            assert self.target_mean is not None and self.target_std is not None  # guaranteed by validation
            return (labels - self.target_mean) / self.target_std
        return labels

    def _inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self._transforms():
            return values
        if self.target_transform == "log1p":
            return np.expm1(values)
        if self.target_transform == "standardize":
            assert self.target_mean is not None and self.target_std is not None  # guaranteed by validation
            return values * self.target_std + self.target_mean
        return values

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map predictions back to the original target space.

        The public counterpart of the transform applied during training, for
        callers that score checkpoint predictions outside the training loop.
        """
        return self._inverse_transform(values)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.objective == "classification":
            return F.cross_entropy(logits, labels)
        return F.mse_loss(logits, self._forward_transform(labels))

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        if self.objective == "classification":
            return logits.argmax(dim=-1)
        return logits

    def epoch_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        if self.objective == "classification":
            accuracy = float((preds == labels).mean()) if len(labels) else 0.0
            return {"accuracy": accuracy}
        preds = self._inverse_transform(preds)
        errors = preds - labels
        mae = float(np.abs(errors).mean()) if len(labels) else 0.0
        mse = float((errors**2).mean()) if len(labels) else 0.0
        var = float(((labels - labels.mean()) ** 2).mean()) if len(labels) else 0.0
        r2 = 1.0 - mse / var if var > 0 else 0.0
        return {"mae": mae, "mse": mse, "r2": r2}

    @property
    def primary_metric(self) -> tuple[str, str]:
        return ("accuracy", "max") if self.objective == "classification" else ("mae", "min")
