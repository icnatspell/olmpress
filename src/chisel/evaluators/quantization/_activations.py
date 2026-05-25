"""Forward-hook based intermediate-activation capture for PyTorch models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Self

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from torch import Tensor, nn


def _first_tensor(output: Any) -> Tensor | None:
    """Return the first ``Tensor`` reachable in a module's output, or ``None``."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for item in output.values():
            if isinstance(item, torch.Tensor):
                return item
    for attr in ("last_hidden_state", "logits", "hidden_states"):
        candidate = getattr(output, attr, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    return None


class ActivationCollector:
    """Capture per-module activations during a forward pass."""

    def __init__(
        self,
        model: nn.Module,
        names: Iterable[str],
        *,
        detach: bool = True,
        cpu: bool = False,
    ) -> None:
        """Bind to ``model`` and prepare to capture the listed module outputs."""
        self._model = model
        self._names = tuple(names)
        self._detach = detach
        self._cpu = cpu
        self._captures: dict[str, Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def captures(self) -> dict[str, Tensor]:
        """Captured ``{name: tensor}`` from the most recent forward pass."""
        return self._captures

    def _make_hook(self, name: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                return
            captured = tensor.detach() if self._detach else tensor
            if self._cpu:
                captured = captured.to("cpu")
            self._captures[name] = captured

        return hook

    def __enter__(self) -> Self:
        """Register forward hooks on the requested modules."""
        named = dict(self._model.named_modules())
        missing = [n for n in self._names if n not in named]
        if missing:
            preview = ", ".join(missing[:5])
            more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            msg = f"ActivationCollector: unknown module name(s): {preview}{more}"
            raise KeyError(msg)
        self._captures = {}
        for name in self._names:
            handle = named[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        """Remove every forward hook registered by ``__enter__``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@contextmanager
def capture(
    model: nn.Module,
    names: Iterable[str],
    *,
    detach: bool = True,
    cpu: bool = False,
) -> Iterator[dict[str, Tensor]]:
    """Yield the captures dict from an :class:`ActivationCollector`."""
    collector = ActivationCollector(model, names, detach=detach, cpu=cpu)
    with collector:
        yield collector.captures
