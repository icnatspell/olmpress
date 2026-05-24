"""Mean-squared error and relative L2 error."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def mse(reference: Tensor, target: Tensor) -> Tensor:
    """Return the mean squared error between two equal-shaped tensors."""
    if reference.shape != target.shape:
        msg = f"mse: shape mismatch {tuple(reference.shape)} vs {tuple(target.shape)}"
        raise ValueError(msg)
    return (reference.to(torch.float32) - target.to(torch.float32)).pow(2).mean()


def relative_l2(reference: Tensor, target: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Return ``||reference - target||_2 / (||reference||_2 + eps)``."""
    if reference.shape != target.shape:
        msg = f"relative_l2: shape mismatch {tuple(reference.shape)} vs {tuple(target.shape)}"
        raise ValueError(msg)
    ref = reference.to(torch.float32)
    tgt = target.to(torch.float32)
    return torch.linalg.vector_norm(ref - tgt) / (torch.linalg.vector_norm(ref) + eps)
