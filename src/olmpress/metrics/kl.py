"""KL divergence on logit distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional

if TYPE_CHECKING:
    from torch import Tensor


Reduction = Literal["mean", "batchmean", "sum", "none"]


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
