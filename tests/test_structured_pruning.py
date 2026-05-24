"""Tests for the TorchPruningPass structured pruning pass."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import torch
from olive.hardware.accelerator import AcceleratorSpec, Device, ExecutionProvider
from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import Pass
from torch import nn

from olmpress.passes.structured_pruning import (
    TorchPruningPass,
    _resolve_ignored_layers,
    prune_model,
)

# ---------------------------------------------------------------------------
# Helpers


class TinyMLP(nn.Module):
    """3-layer MLP: input -> hidden -> hidden -> output."""

    def __init__(self, in_dim: int = 16, hidden: int = 32, out_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _example_input(batch: int = 2, in_dim: int = 16) -> torch.Tensor:
    return torch.randn(batch, in_dim)


# ---------------------------------------------------------------------------
# Unit tests for prune_model()


def test_prune_model_reduces_param_count():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5)
    assert _param_count(model) < before


def test_prune_model_forward_still_works():
    model = TinyMLP()
    # Ignore the head so its output dim is preserved.
    prune_model(model, _example_input(), pruning_ratio=0.5, ignored_layers=[model.head])
    out = model(_example_input())
    assert out.shape[-1] == 8  # output dim unchanged


def test_prune_model_ignored_layer_unchanged():
    model = TinyMLP(hidden=32)
    head_weight_before = model.head.weight.data.clone()
    prune_model(model, _example_input(), pruning_ratio=0.5, ignored_layers=[model.head])
    assert model.head.weight.shape[0] == head_weight_before.shape[0]


def test_prune_model_higher_ratio_gives_smaller_model():
    m_low = TinyMLP()
    m_high = TinyMLP()
    prune_model(m_low, _example_input(), pruning_ratio=0.25)
    prune_model(m_high, _example_input(), pruning_ratio=0.75)
    assert _param_count(m_low) > _param_count(m_high)


def test_prune_model_iterative_steps():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5, iterative_steps=3)
    assert _param_count(model) < before


def test_prune_model_round_to():
    model = TinyMLP(hidden=32)
    prune_model(model, _example_input(), pruning_ratio=0.5, round_to=8)
    assert model.fc1.weight.shape[0] % 8 == 0


def test_prune_model_random_importance():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5, importance="random")
    assert _param_count(model) < before


def test_prune_model_unknown_importance_raises():
    model = TinyMLP()
    with pytest.raises(ValueError, match="Unknown importance"):
        prune_model(model, _example_input(), pruning_ratio=0.5, importance="not_real")


# ---------------------------------------------------------------------------
# Unit tests for _resolve_ignored_layers


def test_resolve_ignored_layers_valid():
    model = TinyMLP()
    result = _resolve_ignored_layers(model, ["head"])
    assert result == [model.head]


def test_resolve_ignored_layers_missing_warns(caplog: pytest.LogCaptureFixture):
    model = TinyMLP()
    with caplog.at_level(logging.WARNING):
        result = _resolve_ignored_layers(model, ["does_not_exist"])
    assert result == []
    assert "does_not_exist" in caplog.text


# ---------------------------------------------------------------------------
# Pass registration


def test_pass_is_registered_in_olive():
    assert "torchpruningpass" in Pass.registry


# ---------------------------------------------------------------------------
# Integration: TorchPruningPass through Olive PyTorchModelHandler


def test_pass_runs_on_pytorch_handler():
    model = TinyMLP()
    before = _param_count(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use model_loader so Olive returns the plain nn.Module (required for pruning).
        def _loader(_path: str) -> nn.Module:
            return model

        def _dummy_inputs(_handler: PyTorchModelHandler) -> torch.Tensor:
            return _example_input()

        handler = PyTorchModelHandler(
            model_loader=_loader,
            dummy_inputs_func=_dummy_inputs,
        )

        accel = AcceleratorSpec(Device.CPU, ExecutionProvider.CPUExecutionProvider)
        cfg_class, _ = TorchPruningPass.get_config_class(accel)
        cfg = cfg_class(pruning_ratio=0.5)
        pass_obj = TorchPruningPass(accel, cfg)  # pyrefly: ignore[bad-argument-type]

        out_path = Path(tmpdir) / "pruned"
        out_handler = pass_obj.run(handler, str(out_path))

        pruned: nn.Module = out_handler.load_model()  # type: ignore[assignment]
        assert _param_count(pruned) < before
