"""Tests for runtime patches applied to olive-ai."""

from __future__ import annotations

import chisel  # noqa: F401  (importing applies the patches)


class _Dummy:
    """Minimal stand-in for an object whose .model_dump() should be passed through."""

    def model_dump(self) -> dict:
        return {"ok": True}


def _get_serializer(cls_name: str, key: str):
    from olive.evaluator.metric import Metric, SubMetric

    cls = {"Metric": Metric, "SubMetric": SubMetric}[cls_name]
    return cls.__pydantic_decorators__.field_serializers[key].func


def test_metric_serialize_backend_handles_none():
    fn = _get_serializer("Metric", "serialize_backend")
    assert fn(None, None) is None


def test_metric_serialize_backend_passes_string_through():
    fn = _get_serializer("Metric", "serialize_backend")
    assert fn(None, "huggingface_metrics") == "huggingface_metrics"


def test_metric_serialize_backend_calls_model_dump_for_objects():
    fn = _get_serializer("Metric", "serialize_backend")
    assert fn(None, _Dummy()) == {"ok": True}


def test_sub_metric_serialize_metric_config_handles_none():
    fn = _get_serializer("SubMetric", "serialize_metric_config")
    assert fn(None, None) is None


def test_sub_metric_serialize_metric_config_calls_model_dump_for_objects():
    fn = _get_serializer("SubMetric", "serialize_metric_config")
    assert fn(None, _Dummy()) == {"ok": True}
