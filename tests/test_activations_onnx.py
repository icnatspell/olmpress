"""Tests for chisel.activations_onnx using a tiny hand-built ONNX graph."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from chisel.evaluators.quantization._activations_onnx import (
    capture_onnx,
    list_intermediate_tensors,
)


def _build_tiny_graph() -> onnx.ModelProto:
    """Build a 3-op model: x -> Add(x, w1) -> Relu -> Mul(_, w2) -> y.

    Intermediate tensors: 'add_out', 'relu_out'. Graph output: 'y'.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    w1 = numpy_helper.from_array(np.ones((2, 3), dtype=np.float32), name="w1")
    w2 = numpy_helper.from_array(np.full((2, 3), 2.0, dtype=np.float32), name="w2")

    nodes = [
        helper.make_node("Add", ["x", "w1"], ["add_out"], name="add"),
        helper.make_node("Relu", ["add_out"], ["relu_out"], name="relu"),
        helper.make_node("Mul", ["relu_out", "w2"], ["y"], name="mul"),
    ]
    graph = helper.make_graph(nodes, "tiny", [x], [y], initializer=[w1, w2])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def test_list_intermediate_tensors_finds_internal_outputs():
    model = _build_tiny_graph()
    intermediates = list_intermediate_tensors(model)
    assert "add_out" in intermediates
    assert "relu_out" in intermediates
    # 'y' is a graph output, not intermediate
    assert "y" not in intermediates
    # Initializers are not node outputs
    assert "w1" not in intermediates


def test_capture_returns_requested_tensors():
    model = _build_tiny_graph()
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    caps = capture_onnx(model, ["add_out", "relu_out"], {"x": x})
    np.testing.assert_array_equal(caps["add_out"], x + 1.0)
    np.testing.assert_array_equal(caps["relu_out"], np.maximum(x + 1.0, 0.0))


def test_capture_does_not_alter_existing_graph_output():
    """Adding extra outputs shouldn't change the value of pre-existing outputs."""
    model = _build_tiny_graph()
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]], dtype=np.float32)
    caps = capture_onnx(model, ["y"], {"x": x})
    expected = np.maximum(x + 1.0, 0.0) * 2.0
    np.testing.assert_allclose(caps["y"], expected)


def test_capture_unknown_tensor_raises():
    model = _build_tiny_graph()
    x = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(KeyError, match="unknown tensor name"):
        capture_onnx(model, ["does_not_exist"], {"x": x})


def test_capture_preserves_request_order():
    model = _build_tiny_graph()
    x = np.zeros((2, 3), dtype=np.float32)
    caps = capture_onnx(model, ["relu_out", "add_out"], {"x": x})
    assert list(caps) == ["relu_out", "add_out"]


def test_capture_dedupes_repeated_names():
    model = _build_tiny_graph()
    x = np.zeros((2, 3), dtype=np.float32)
    caps = capture_onnx(model, ["add_out", "add_out"], {"x": x})
    assert list(caps) == ["add_out"]


def test_original_model_not_mutated():
    """Augmenting the graph for capture must not touch the input model in place."""
    model = _build_tiny_graph()
    outputs_before = {out.name for out in model.graph.output}
    capture_onnx(model, ["add_out"], {"x": np.zeros((2, 3), dtype=np.float32)})
    outputs_after = {out.name for out in model.graph.output}
    assert outputs_before == outputs_after
