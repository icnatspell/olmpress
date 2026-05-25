"""Olive model script for microsoft/resnet-50 (HuggingFace Hub).

The HF model returns an ImageClassifierOutput dataclass. This module wraps
it with ``return_dict=False`` so ``forward`` returns a plain logits tensor
that torch-pruning can trace without changes.

Olive references these functions from the YAML via ``model_loader`` and
``dummy_inputs_func``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from olive.model import PyTorchModelHandler

MODEL_ID = "microsoft/resnet-50"


class _LogitsWrapper(nn.Module):
    """Thin wrapper: HF ResNet → plain logits tensor."""

    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModelForImageClassification

        self._model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # return_dict=False → tuple; logits is the first (and only) element.
        out = self._model(pixel_values, return_dict=False)
        return out[0]


def load_model(_model_path: str | None) -> nn.Module:
    """Return microsoft/resnet-50 wrapped to output raw logits."""
    return _LogitsWrapper().eval()


def get_dummy_inputs(_handler: PyTorchModelHandler) -> torch.Tensor:
    """Return a random (1, 3, 224, 224) ImageNet-sized input tensor."""
    return torch.randn(1, 3, 224, 224)
