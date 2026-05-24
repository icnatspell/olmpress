"""Custom Olive optimization passes."""

from olmpress.passes.structured_pruning import TorchPruningPass, prune_model

__all__ = ["TorchPruningPass", "prune_model"]
