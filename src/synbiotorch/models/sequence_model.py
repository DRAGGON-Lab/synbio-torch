"""A sequence model: backbone + pooler + task head."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from synbiotorch.models.conv import SequenceBackbone
from synbiotorch.models.pooling import MeanPooler


class SequenceModel(nn.Module):
    """Composes a sequence backbone, a pooler, and a task head.

    The backbone is either a HuggingFace encoder — attached directly, so its
    ``state_dict`` keys stay ``backbone.*`` — or a ``SequenceBackbone``, which
    additionally returns the attention mask for its own output length because a
    convolution or pooling stack changes it.

    Poolers hold no parameters, so which one is used does not affect the
    checkpoint.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        *,
        pooler: nn.Module | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler if pooler is not None else MeanPooler()
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the backbone and return ``(hidden_state, mask_for_that_hidden_state)``."""
        if isinstance(self.backbone, SequenceBackbone):
            return self.backbone(input_ids, attention_mask)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        # A HuggingFace encoder preserves length, so the input mask still applies.
        return hidden_state, attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            hidden_state, mask = self._encode(input_ids, attention_mask)
        return self.head(self.pooler(hidden_state, mask))
