"""Cosine similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


def cosine_similarity(reference: Tensor, target: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Return the cosine similarity of two equal-shaped tensors, flattened."""
    if reference.shape != target.shape:
        msg = f"cosine_similarity: shape mismatch {tuple(reference.shape)} vs {tuple(target.shape)}"
        raise ValueError(msg)
    ref = reference.to(torch.float32).reshape(-1)
    tgt = target.to(torch.float32).reshape(-1)
    return torch.dot(ref, tgt) / (
        (torch.linalg.vector_norm(ref) + eps) * (torch.linalg.vector_norm(tgt) + eps)
    )
