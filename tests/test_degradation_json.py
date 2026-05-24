"""JSON / workflow integration tests for DegradationEvaluator.

These cover the same shape of construction Olive's workflow runner uses when
parsing a config file: dict-shaped ``reference_model``, dict-shaped ``inputs``,
and target models wrapped in real :class:`PyTorchModelHandler` /
:class:`ONNXModelHandler` instances (rather than raw modules).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import pytest
import torch
from olive.evaluator.metric import Metric, MetricType
from olive.model.handler.onnx import ONNXModelHandler
from olive.model.handler.pytorch import PyTorchModelHandler
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from olmpress.evaluators.degradation import DegradationEvaluator, make_inputs_loader

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult


def _val(res: MetricResult, name: str, sub: str) -> float:
    v = res.get_value(name, sub)
    assert v is not None
    return float(v)


def _make_metric(names: list[str]) -> Metric:
    return Metric(
        name="deg",
        type=MetricType.CUSTOM,
        sub_types=[{"name": n} for n in names],
    )


# ---------------------------------------------------------------------------
# make_inputs_loader


def test_make_inputs_loader_integer_uniform():
    loader = make_inputs_loader(
        {"input_ids": {"shape": [2, 4], "dtype": "long", "low": 0, "high": 16}}
    )
    out = loader()
    assert isinstance(out["input_ids"], torch.Tensor)
    assert out["input_ids"].shape == (2, 4)
    assert out["input_ids"].dtype == torch.int64
    assert int(out["input_ids"].min()) >= 0
    assert int(out["input_ids"].max()) < 16


def test_make_inputs_loader_float_normal():
    loader = make_inputs_loader({"x": {"shape": [3, 5], "dtype": "float32"}})
    out = loader()
    assert out["x"].dtype == torch.float32
    assert out["x"].shape == (3, 5)


def test_make_inputs_loader_fixed_value():
    loader = make_inputs_loader({"x": {"shape": [2, 2], "value": 1.5, "dtype": "float32"}})
    out = loader()
    assert torch.equal(out["x"], torch.full((2, 2), 1.5))


def test_make_inputs_loader_numpy_framework():
    loader = make_inputs_loader(
        {"input": {"shape": [2, 4], "dtype": "float32", "framework": "numpy"}}
    )
    out = loader()
    assert isinstance(out["input"], np.ndarray)
    assert out["input"].dtype == np.float32
    assert out["input"].shape == (2, 4)


def test_make_inputs_loader_is_deterministic_via_seed():
    spec = {"x": {"shape": [4], "dtype": "float32", "seed": 42}}
    a = make_inputs_loader(spec)()["x"]
    b = make_inputs_loader(spec)()["x"]
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Dict-config construction (the JSON path)


class _Toy(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(16, dim)
        self.lm_head = nn.Linear(dim, 16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed(input_ids))


def _write_toy_loader_script(tmp: Path) -> Path:
    """Write a Python module exporting ``load(path)`` returning a deterministic ``_Toy()``."""
    script = tmp / "toy_loader.py"
    script.write_text(
        """
import torch
from torch import nn


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16)

    def forward(self, input_ids):
        return self.lm_head(self.embed(input_ids))


def load(_model_path):
    torch.manual_seed(0)
    return Toy()
"""
    )
    return script


def test_reference_model_dict_resolves_via_model_config(tmp_path: Path):
    """A dict-shaped ``reference_model`` with ``model_script`` + ``model_loader`` should resolve."""
    script = _write_toy_loader_script(tmp_path)
    ev = DegradationEvaluator(
        reference_model={
            "type": "PyTorchModel",
            "config": {
                "model_path": str(script),  # nominal; loader ignores it
                "model_script": str(script),
                "model_loader": "load",
            },
        },
        inputs={"input_ids": {"shape": [2, 4], "dtype": "long", "low": 0, "high": 16}},
    )
    # The target uses the *same* loader → identical weights.
    torch.manual_seed(0)
    target = _Toy()  # same seed as the script ⇒ same parameters
    res = ev.evaluate(target, [_make_metric(["mse_mean"])])
    assert _val(res, "deg", "mse_mean") == pytest.approx(0.0, abs=1e-8)


def test_pytorch_model_handler_target_works():
    """Target wrapped in a real PyTorchModelHandler (callable model_loader) is unwrapped."""
    torch.manual_seed(0)
    reference = _Toy()
    torch.manual_seed(0)
    target = _Toy()
    handler = PyTorchModelHandler(model_loader=lambda *_a, **_k: target)
    ev = DegradationEvaluator(
        reference_model=lambda: reference,
        inputs={"input_ids": {"shape": [2, 4], "dtype": "long", "low": 0, "high": 16}},
    )
    res = ev.evaluate(handler, [_make_metric(["mse_mean"])])
    assert _val(res, "deg", "mse_mean") == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# ONNX handler integration


def _save_tiny_onnx(tmp: Path, w1_val: float = 1.0) -> Path:
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
    path = tmp / f"tiny_{w1_val}.onnx"
    onnx.save(model, str(path))
    return path


def test_onnx_handler_target_works(tmp_path: Path):
    ref_path = _save_tiny_onnx(tmp_path, w1_val=1.0)
    tgt_path = _save_tiny_onnx(tmp_path, w1_val=1.0)
    handler = ONNXModelHandler(model_path=str(tgt_path))

    ev = DegradationEvaluator(
        reference_model={"type": "ONNXModel", "config": {"model_path": str(ref_path)}},
        inputs={
            "input": {
                "shape": [2, 4],
                "dtype": "float32",
                "framework": "numpy",
                "value": 0.5,
            }
        },
        logits_layer="lm_head",
    )
    res = ev.evaluate(
        handler,
        [_make_metric(["mse_mean", "sqnr_mean"])],
        execution_providers=["CPUExecutionProvider"],
    )
    assert _val(res, "deg", "mse_mean") == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Example workflow JSON


def test_example_workflow_json_is_valid_and_references_us():
    """The shipped example workflow JSON should parse and reference our evaluator."""
    repo_root = Path(__file__).resolve().parent.parent
    workflow_path = repo_root / "examples" / "qwen_mixed_with_degradation.json"
    assert workflow_path.exists(), f"missing example workflow at {workflow_path}"
    data = json.loads(workflow_path.read_text())
    # Surface checks: our evaluator is referenced
    evaluators = data.get("evaluators", {})
    degradation_evaluators = [
        ev for ev in evaluators.values() if ev.get("type") == "olmpress_degradation"
    ]
    assert degradation_evaluators, "example workflow should reference olmpress_degradation"
    # Each one must specify reference_model and inputs (the two required knobs)
    for ev in degradation_evaluators:
        assert "reference_model" in ev
        assert "inputs" in ev


def test_evaluator_constructible_from_workflow_json_evaluator_block(tmp_path: Path):
    """Mimic Olive parsing one evaluator entry from a workflow JSON and constructing us."""
    script = _write_toy_loader_script(tmp_path)
    evaluator_block: dict[str, Any] = {
        "type": "olmpress_degradation",
        "reference_model": {
            "type": "PyTorchModel",
            "config": {
                "model_path": str(script),
                "model_script": str(script),
                "model_loader": "load",
            },
        },
        "inputs": {"input_ids": {"shape": [2, 4], "dtype": "long", "low": 0, "high": 16}},
        "view": "linears",
        "metrics": [
            {"name": "deg", "type": "custom", "sub_types": [{"name": "sqnr_mean"}]},
        ],
    }
    # This is exactly how Olive instantiates a registered evaluator: it strips
    # ``type`` and ``metrics`` from the block, then calls ``cls(**rest)`` and
    # passes the parsed metrics into ``evaluate()``.
    kwargs: dict[str, Any] = {
        k: v for k, v in evaluator_block.items() if k not in ("type", "metrics")
    }
    ev = DegradationEvaluator(**kwargs)
    metrics = [Metric(**{**m, "type": MetricType(m["type"])}) for m in evaluator_block["metrics"]]
    torch.manual_seed(0)
    target = _Toy()
    res = ev.evaluate(target, metrics)
    assert _val(res, "deg", "sqnr_mean") > 100  # identical → very high SQNR
