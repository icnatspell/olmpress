"""Custom Olive optimization passes."""

from chisel.passes.pytorch.structured_pruning import TorchPruningPass, prune_model

__all__ = ["TorchPruningPass", "prune_model"]
