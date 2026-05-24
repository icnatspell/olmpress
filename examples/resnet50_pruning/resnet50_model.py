"""Olive model script for pretrained ResNet-50.

Olive references these functions from the YAML via ``model_loader`` and
``dummy_inputs_func``.  Both signatures match what PyTorchModelHandler expects:

* ``load_model(model_path)``  — model_path is None for script-loaded models
* ``get_dummy_inputs(handler)`` — handler is the PyTorchModelHandler instance
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchvision.models as tvm

if TYPE_CHECKING:
    from olive.model import PyTorchModelHandler


def load_model(_model_path: str | None) -> torch.nn.Module:
    """Return a pretrained ResNet-50 (ImageNet-1k V1 weights)."""
    return tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1).eval()


def get_dummy_inputs(_handler: PyTorchModelHandler) -> torch.Tensor:
    """Return a random (1, 3, 224, 224) ImageNet-sized input tensor."""
    return torch.randn(1, 3, 224, 224)
