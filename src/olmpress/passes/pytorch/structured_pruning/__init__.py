"""Structured channel pruning pass for PyTorch models."""

from olmpress.passes.pytorch.structured_pruning.pruning_pass import TorchPruningPass
from olmpress.passes.pytorch.structured_pruning.utils import prune_model

__all__ = ["TorchPruningPass", "prune_model"]
