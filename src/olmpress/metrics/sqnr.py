"""Signal-to-Quantization-Noise Ratio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


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
