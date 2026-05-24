"""Runtime patches for Olive bugs."""

from __future__ import annotations


def apply() -> None:
    """Apply all patches. Safe to call multiple times."""
    _patch_metric_serialize_backend()
    _patch_sub_metric_serialize_metric_config()


def _patch_metric_serialize_backend() -> None:
    # Olive bug: Metric.serialize_backend does not guard against None,
    # but validate_backend always sets backend=None for MetricType.CUSTOM.
    from olive.evaluator.metric import Metric  # noqa: PLC0415

    original_func = Metric.__pydantic_decorators__.field_serializers["serialize_backend"].func

    def _fixed(self: object, backend: object) -> object:  # noqa: ARG001
        if backend is None or isinstance(backend, str):
            return backend
        return backend.model_dump()  # type: ignore[union-attr]

    original_func.__code__ = _fixed.__code__


def _patch_sub_metric_serialize_metric_config() -> None:
    # Olive bug: SubMetric.serialize_metric_config does not guard against None,
    # but metric_config is Optional and can be None for custom sub-types.
    from olive.evaluator.metric import SubMetric  # noqa: PLC0415

    original_func = SubMetric.__pydantic_decorators__.field_serializers[
        "serialize_metric_config"
    ].func

    def _fixed(self: object, metric_config: object) -> object:  # noqa: ARG001
        if metric_config is None:
            return None
        return metric_config.model_dump()  # type: ignore[union-attr]

    original_func.__code__ = _fixed.__code__
