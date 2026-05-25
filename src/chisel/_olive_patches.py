"""Runtime patches for known olive-ai bugs.

Each patch documents:
  - which olive-ai versions are affected
  - what the upstream bug is
  - the upstream issue/PR (if filed) so we can remove the patch once fixed

When bumping the ``olive-ai`` floor in ``pyproject.toml`` past an affected range,
delete the corresponding patch here.
"""

from __future__ import annotations


def apply() -> None:
    """Apply all patches. Safe to call multiple times."""
    _patch_metric_serialize_backend()
    _patch_sub_metric_serialize_metric_config()


def _patch_metric_serialize_backend() -> None:
    """Fix Metric.serialize_backend for MetricType.CUSTOM (backend=None).

    Affected versions: olive-ai 0.12.x (current floor).
    Bug: ``Metric.serialize_backend`` unconditionally calls ``backend.model_dump()``,
    but ``validate_backend`` always sets ``backend=None`` for ``MetricType.CUSTOM``.
    Upstream: not yet filed — TODO file at microsoft/Olive.

    We replace the function code via ``__code__`` swap because Pydantic has already
    captured the serializer at class-creation time, so reassigning ``.func`` has no
    effect on the bound descriptor.
    """
    from olive.evaluator.metric import Metric

    original_func = Metric.__pydantic_decorators__.field_serializers["serialize_backend"].func

    def _fixed(self: object, backend: object) -> object:
        if backend is None or isinstance(backend, str):
            return backend
        return backend.model_dump()  # type: ignore[union-attr]

    original_func.__code__ = _fixed.__code__


def _patch_sub_metric_serialize_metric_config() -> None:
    """Fix SubMetric.serialize_metric_config when metric_config is None.

    Affected versions: olive-ai 0.12.x (current floor).
    Bug: ``SubMetric.serialize_metric_config`` unconditionally calls
    ``metric_config.model_dump()``, but ``metric_config`` is ``Optional`` and is
    ``None`` for custom sub-types.
    Upstream: not yet filed — TODO file at microsoft/Olive.
    """
    from olive.evaluator.metric import SubMetric

    original_func = SubMetric.__pydantic_decorators__.field_serializers[
        "serialize_metric_config"
    ].func

    def _fixed(self: object, metric_config: object) -> object:
        if metric_config is None:
            return None
        return metric_config.model_dump()  # type: ignore[union-attr]

    original_func.__code__ = _fixed.__code__
