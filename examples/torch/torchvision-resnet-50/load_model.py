"""Olive model script for torchvision ResNet50.

Loads torchvision.models.resnet50 with IMAGENET1K_V2 weights.
The model already outputs raw logits so no wrapper is needed.

Olive references these functions from workflow.yaml via ``model_loader``
and ``dummy_inputs_func``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torchvision.models import ResNet50_Weights, resnet50

if TYPE_CHECKING:
    from olive.model import PyTorchModelHandler


def load_model(_model_path: str | None) -> torch.nn.Module:
    """Return torchvision ResNet50 with IMAGENET1K_V2 pretrained weights."""
    return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()


def get_dummy_inputs(_handler: PyTorchModelHandler) -> torch.Tensor:
    """Return a random (1, 3, 224, 224) ImageNet-sized input tensor."""
    return torch.randn(1, 3, 224, 224)
