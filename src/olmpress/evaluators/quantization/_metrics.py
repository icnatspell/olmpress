"""Metrics for measuring quantization-induced degradation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional

if TYPE_CHECKING:
    from torch import Tensor

Reduction = Literal["mean", "batchmean", "sum", "none"]


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


def kl_divergence(
    reference_logits: Tensor,
    target_logits: Tensor,
    *,
    dim: int = -1,
    reduction: Reduction = "batchmean",
    temperature: float = 1.0,
) -> Tensor:
    """Return ``KL(reference || target)`` over the distribution axis ``dim``."""
    if reference_logits.shape != target_logits.shape:
        msg = (
            f"kl_divergence: shape mismatch "
            f"{tuple(reference_logits.shape)} vs {tuple(target_logits.shape)}"
        )
        raise ValueError(msg)
    if temperature <= 0:
        msg = f"kl_divergence: temperature must be > 0, got {temperature}"
        raise ValueError(msg)

    ref = reference_logits.to(torch.float32) / temperature
    tgt = target_logits.to(torch.float32) / temperature
    log_p = functional.log_softmax(ref, dim=dim)
    log_q = functional.log_softmax(tgt, dim=dim)
    return functional.kl_div(log_q, log_p, log_target=True, reduction=reduction)


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


def sqnr(reference: Tensor, target: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Return ``10 * log10(||reference||^2 / ||reference - target||^2)`` in dB."""
    if reference.shape != target.shape:
        msg = f"sqnr: shape mismatch {tuple(reference.shape)} vs {tuple(target.shape)}"
        raise ValueError(msg)
    ref = reference.to(torch.float32)
    tgt = target.to(torch.float32)
    signal_power = ref.pow(2).sum()
    noise_power = (ref - tgt).pow(2).sum() + eps
    return 10.0 * torch.log10(signal_power / noise_power)


__all__ = [
    "cosine_similarity",
    "kl_divergence",
    "mse",
    "relative_l2",
    "sqnr",
]
