"""Tests for chisel.activations (PyTorch forward-hook capture)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from chisel.evaluators.quantization._activations import ActivationCollector, capture


class TupleBlock(nn.Module):
    """A block that returns (out, aux) — like attention layers in HF models."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.lin(x)
        return out, out.detach()


class DictBlock(nn.Module):
    """A block that returns a dict — like HF ModelOutput subclasses."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.lin(x)
        return {"last_hidden_state": out}


class Model(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.tup = TupleBlock(dim)
        self.dct = DictBlock(dim)
        self.lm_head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x, _ = self.tup(x)
        x = self.dct(x)["last_hidden_state"]
        return self.lm_head(x)


def _model_and_input():
    torch.manual_seed(0)
    m = Model()
    x = torch.randn(2, 4)
    return m, x


def test_collector_captures_requested_names():
    model, x = _model_and_input()
    names = ["layers.0", "layers.2", "lm_head"]
    with ActivationCollector(model, names) as col:
        model(x)
    assert set(col.captures) == set(names)
    for tensor in col.captures.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 4)


def test_collector_unknown_name_raises():
    model, _ = _model_and_input()
    with (
        pytest.raises(KeyError, match="unknown module name"),
        ActivationCollector(model, ["nope"]),
    ):
        pass


def test_capture_helper_yields_dict():
    model, x = _model_and_input()
    with capture(model, ["embed", "lm_head"]) as caps:
        model(x)
    assert set(caps) == {"embed", "lm_head"}


def test_captures_are_detached_by_default():
    model, x = _model_and_input()
    with capture(model, ["lm_head"]) as caps:
        model(x)
    assert caps["lm_head"].requires_grad is False


def test_tuple_output_takes_first_tensor():
    model, x = _model_and_input()
    with capture(model, ["tup"]) as caps:
        model(x)
    assert caps["tup"].shape == (2, 4)


def test_dict_output_extracts_tensor():
    model, x = _model_and_input()
    with capture(model, ["dct"]) as caps:
        model(x)
    assert caps["dct"].shape == (2, 4)


def test_hooks_removed_after_context_exit():
    model, _ = _model_and_input()
    layer = model.layers[0]
    before = len(layer._forward_hooks)
    with capture(model, ["layers.0"]):
        pass
    assert len(layer._forward_hooks) == before


def test_same_input_gives_same_captures_for_identical_models():
    torch.manual_seed(0)
    a = Model()
    torch.manual_seed(0)
    b = Model()  # identical weights via fixed seed
    x = torch.randn(2, 4)
    with capture(a, ["lm_head"]) as ca:
        a(x)
    with capture(b, ["lm_head"]) as cb:
        b(x)
    assert torch.allclose(ca["lm_head"], cb["lm_head"])


def test_cpu_flag_moves_capture_to_cpu():
    model, x = _model_and_input()
    with capture(model, ["lm_head"], cpu=True) as caps:
        model(x)
    assert caps["lm_head"].device.type == "cpu"


def test_no_capture_when_output_has_no_tensor():
    class NoTensor(nn.Module):
        def forward(self, x: torch.Tensor) -> str:
            del x
            return "no tensors here"

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = NoTensor()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.inner(x)
            return x

    m = Wrap()
    x = torch.randn(2, 4)
    with capture(m, ["inner"]) as caps:
        m(x)
    assert caps == {}
