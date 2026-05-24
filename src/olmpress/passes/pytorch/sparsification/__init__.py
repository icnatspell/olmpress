"""Sparsification passes (pruning, etc.) for PyTorch models."""

from olmpress.passes.pytorch.structured_pruning import TorchPruningPass, prune_model

__all__ = ["TorchPruningPass", "prune_model"]
