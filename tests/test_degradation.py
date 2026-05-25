"""End-to-end tests for the QuantErrorEvaluator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
import torch
from olive.evaluator.metric import Metric, MetricType
from olive.evaluator.registry import Registry
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from chisel.evaluators.quantization.evaluator import QuantErrorEvaluator, supported_sub_types

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult


def _val(res: MetricResult, name: str, sub: str) -> float:
    """Return ``res.get_value(...)`` asserting it is not ``None`` (narrows type)."""
    v = res.get_value(name, sub)
    assert v is not None, f"missing metric {name}-{sub}"
    return float(v)


# ---------------------------------------------------------------------------
# Fixtures


class TinyModel(nn.Module):
    """A small toy model whose forward returns logits via ``lm_head``."""

    def __init__(self, dim: int = 8, n_layers: int = 2, vocab: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def _identical_models():
    torch.manual_seed(0)
    a = TinyModel()
    torch.manual_seed(0)
    b = TinyModel()
    return a, b


def _noised_pair(noise: float = 0.01):
    torch.manual_seed(0)
    a = TinyModel()
    b = TinyModel()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        for p in b.parameters():
            p.add_(noise * torch.randn_like(p))
    return a, b


def _inputs():
    return {"input_ids": torch.randint(0, 16, (2, 4))}


def _make_metric(sub_type_names: list[str]) -> Metric:
    return Metric(
        name="degradation",
        type=MetricType.CUSTOM,
        sub_types=[{"name": n} for n in sub_type_names],
    )


# ---------------------------------------------------------------------------
# Registry


def test_evaluator_is_registered_under_expected_name():
    assert Registry.get("chisel_quant_error") is QuantErrorEvaluator


def test_supported_sub_types_includes_expected():
    expected = {
        "sqnr_mean",
        "sqnr_min",
        "sqnr_max",
        "sqnr_p50",
        "cosine_mean",
        "cosine_min",
        "mse_mean",
        "mse_max",
        "relative_l2_mean",
        "relative_l2_max",
        "kl",
    }
    assert expected.issubset(set(supported_sub_types()))


# ---------------------------------------------------------------------------
# PyTorch path


def test_identical_models_yield_perfect_scores():
    ref, tgt = _identical_models()
    ev = QuantErrorEvaluator(
        reference_model=lambda: ref,
        inputs=_inputs,
    )
    metric = _make_metric(["sqnr_mean", "cosine_mean", "mse_mean", "kl"])
    result = ev.evaluate(tgt, [metric])
    assert _val(result, "degradation", "sqnr_mean") > 100  # ~infinite
    assert _val(result, "degradation", "cosine_mean") == pytest.approx(1.0, abs=1e-4)
    assert _val(result, "degradation", "mse_mean") == pytest.approx(0.0, abs=1e-8)
    assert _val(result, "degradation", "kl") == pytest.approx(0.0, abs=1e-6)


def test_more_noise_yields_worse_scores():
    ref, tgt_low = _noised_pair(noise=0.01)
    _, tgt_high = _noised_pair(noise=0.1)
    inputs = _inputs()
    ev_factory = lambda r: QuantErrorEvaluator(
        reference_model=lambda: r,
        inputs=lambda: inputs,
    )
    metric = _make_metric(["sqnr_mean", "mse_mean", "relative_l2_mean", "kl"])
    low = ev_factory(ref).evaluate(tgt_low, [metric])
    high = ev_factory(ref).evaluate(tgt_high, [metric])

    assert _val(low, "degradation", "sqnr_mean") > _val(high, "degradation", "sqnr_mean")
    assert _val(low, "degradation", "mse_mean") < _val(high, "degradation", "mse_mean")
    assert _val(low, "degradation", "relative_l2_mean") < _val(
        high, "degradation", "relative_l2_mean"
    )
    assert _val(low, "degradation", "kl") < _val(high, "degradation", "kl")


def test_sub_type_aggregations_min_max_p50():
    ref, tgt = _noised_pair(noise=0.05)
    ev = QuantErrorEvaluator(reference_model=lambda: ref, inputs=_inputs)
    metric = _make_metric(["sqnr_min", "sqnr_max", "sqnr_p50", "sqnr_mean"])
    res = ev.evaluate(tgt, [metric])
    s_min = _val(res, "degradation", "sqnr_min")
    s_max = _val(res, "degradation", "sqnr_max")
    s_p50 = _val(res, "degradation", "sqnr_p50")
    s_mean = _val(res, "degradation", "sqnr_mean")
    assert s_min <= s_p50 <= s_max
    assert s_min <= s_mean <= s_max


def test_view_linears_only_includes_linear_layers():
    ref, tgt = _noised_pair(noise=0.05)
    ev = QuantErrorEvaluator(reference_model=lambda: ref, inputs=_inputs, view="linears")
    metric = _make_metric(["sqnr_mean"])
    # Should still produce a value — Linear layers exist
    res = ev.evaluate(tgt, [metric])
    assert isinstance(_val(res, "degradation", "sqnr_mean"), float)


def test_unknown_sub_type_raises():
    ref, tgt = _identical_models()
    ev = QuantErrorEvaluator(reference_model=lambda: ref, inputs=_inputs)
    metric = _make_metric(["totally_made_up"])
    with pytest.raises(ValueError, match="unknown sub_type"):
        ev.evaluate(tgt, [metric])


def test_missing_loaders_raises():
    _, tgt = _identical_models()
    ev = QuantErrorEvaluator()  # no loaders
    with pytest.raises(RuntimeError, match="requires both"):
        ev.evaluate(tgt, [_make_metric(["sqnr_mean"])])


def test_pytorch_onnx_cross_framework_requires_dict_spec():
    # Cross-framework path requires a dict spec (for ModelBuilder); a callable has no model_path.
    ref, _ = _identical_models()
    onnx_model = _build_tiny_onnx()
    ev = QuantErrorEvaluator(reference_model=lambda: ref, inputs=_inputs)
    with pytest.raises(RuntimeError, match="dict spec"):
        ev.evaluate(onnx_model, [_make_metric(["sqnr_mean"])])


# ---------------------------------------------------------------------------
# ONNX path


def _build_tiny_onnx(w1_val: float = 1.0) -> onnx.ModelProto:
    """Build an ONNX model computing ``y = Mul(Relu(Add(x, w1)), w2)``."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("lm_head", TensorProto.FLOAT, [2, 4])
    w1 = numpy_helper.from_array(np.full((2, 4), w1_val, dtype=np.float32), name="w1")
    w2 = numpy_helper.from_array(np.full((2, 4), 2.0, dtype=np.float32), name="w2")
    nodes = [
        helper.make_node("Add", ["input", "w1"], ["add_out"], name="add"),
        helper.make_node("Relu", ["add_out"], ["relu_out"], name="relu"),
        helper.make_node("Mul", ["relu_out", "w2"], ["lm_head"], name="mul"),
    ]
    graph = helper.make_graph(nodes, "tiny", [x], [y], initializer=[w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def test_onnx_identical_models_yield_perfect_scores():
    ref = _build_tiny_onnx(w1_val=1.0)
    tgt = _build_tiny_onnx(w1_val=1.0)
    x = np.array([[-1, 0, 1, 2], [3, -2, 0, 1]], dtype=np.float32)
    ev = QuantErrorEvaluator(
        reference_model=lambda: ref,
        inputs=lambda: {"input": x},
        logits_layer="lm_head",
    )
    metric = _make_metric(["sqnr_mean", "mse_mean"])
    res = ev.evaluate(tgt, [metric], execution_providers=["CPUExecutionProvider"])
    assert _val(res, "degradation", "sqnr_mean") > 100
    assert _val(res, "degradation", "mse_mean") == pytest.approx(0.0, abs=1e-8)


def test_onnx_perturbed_models_have_finite_error():
    ref = _build_tiny_onnx(w1_val=1.0)
    tgt = _build_tiny_onnx(w1_val=1.01)
    x = np.array([[-1, 0, 1, 2], [3, -2, 0, 1]], dtype=np.float32)
    ev = QuantErrorEvaluator(
        reference_model=lambda: ref,
        inputs=lambda: {"input": x},
        logits_layer="lm_head",
    )
    metric = _make_metric(["mse_mean", "relative_l2_mean"])
    res = ev.evaluate(tgt, [metric], execution_providers=["CPUExecutionProvider"])
    mse_v = _val(res, "degradation", "mse_mean")
    rl2 = _val(res, "degradation", "relative_l2_mean")
    assert mse_v > 0
    assert rl2 > 0
    assert np.isfinite(mse_v)
    assert np.isfinite(rl2)


# ---------------------------------------------------------------------------
# Composition checks


def test_returns_metric_result_keyed_by_joint_key():
    ref, tgt = _identical_models()
    ev = QuantErrorEvaluator(reference_model=lambda: ref, inputs=_inputs)
    metric = _make_metric(["sqnr_mean", "kl"])
    res = ev.evaluate(tgt, [metric])
    assert set(res.root.keys()) == {"degradation-sqnr_mean", "degradation-kl"}


def test_multiple_metric_groups_are_keyed_separately():
    ref, tgt = _identical_models()
    ev = QuantErrorEvaluator(reference_model=lambda: ref, inputs=_inputs)
    m1 = Metric(name="block_err", type=MetricType.CUSTOM, sub_types=[{"name": "sqnr_mean"}])
    m2 = Metric(name="logits_err", type=MetricType.CUSTOM, sub_types=[{"name": "kl"}])
    res = ev.evaluate(tgt, [m1, m2])
    assert "block_err-sqnr_mean" in res.root
    assert "logits_err-kl" in res.root
