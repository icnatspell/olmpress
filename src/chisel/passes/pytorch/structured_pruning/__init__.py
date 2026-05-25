"""Structured channel pruning pass for PyTorch models."""

from chisel.passes.pytorch.structured_pruning.base import TorchPruningPass
from chisel.passes.pytorch.structured_pruning.utils import prune_model

__all__ = ["TorchPruningPass", "prune_model"]
