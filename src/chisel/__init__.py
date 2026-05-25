"""chisel: model compression toolkit on top of Microsoft Olive."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chisel")
except PackageNotFoundError:  # not installed (e.g. running from source without install)
    __version__ = "0.0.0+unknown"

# Import sub-packages so @Registry.register decorators fire before Olive starts.
from chisel import evaluators as _evaluators  # noqa: F401
from chisel import passes as _passes  # noqa: F401
from chisel._olive_patches import apply as _apply_olive_patches

_apply_olive_patches()

from chisel.cli import main  # noqa: E402  (re-exported for backwards compatibility)

__all__ = ["__version__", "main"]
