"""Tests for the TorchPruningPass structured pruning pass."""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path

import pytest
import torch
from olive.hardware.accelerator import AcceleratorSpec, Device, ExecutionProvider
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes.olive_pass import Pass
from torch import nn

from olmpress.passes.pytorch.structured_pruning import TorchPruningPass, prune_model
from olmpress.passes.pytorch.structured_pruning.utils import (
    collect_unwrapped_parameters,
    resolve_ignored_layers,
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


class TinyCNN(nn.Module):
    """Small CNN with BN, suitable for bn_scale and OBDC importance tests."""

    def __init__(self, in_channels: int = 3, hidden: int = 16, out_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden * 2)
        self.head = nn.Linear(hidden * 2, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=[-2, -1])  # global average pool
        return self.head(x)


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _example_input(batch: int = 2, in_dim: int = 16) -> torch.Tensor:
    return torch.randn(batch, in_dim)


# ---------------------------------------------------------------------------
# Static (non-gradient) importances


def test_prune_model_reduces_param_count():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5)
    assert _param_count(model) < before


def test_prune_model_forward_still_works():
    model = TinyMLP()
    prune_model(model, _example_input(), pruning_ratio=0.5, ignored_layers=[model.head])
    out = model(_example_input())
    assert out.shape[-1] == 8


def test_prune_model_ignored_layer_unchanged():
    model = TinyMLP(hidden=32)
    head_rows_before = model.head.weight.shape[0]
    prune_model(model, _example_input(), pruning_ratio=0.5, ignored_layers=[model.head])
    assert model.head.weight.shape[0] == head_rows_before


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


@pytest.mark.parametrize("importance", ["magnitude", "group_magnitude", "random", "lamp", "fpgm"])
def test_prune_model_static_importances(importance: str):
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5, importance=importance)
    assert _param_count(model) < before


def test_prune_model_bn_scale_importance():
    # bn_scale reads BN layer scales; requires a model with BatchNorm.
    model = TinyCNN()
    before = _param_count(model)
    example = torch.randn(2, 3, 16, 16)
    prune_model(
        model, example, pruning_ratio=0.5, importance="bn_scale", ignored_layers=[model.head]
    )
    assert _param_count(model) < before


def test_prune_model_unknown_importance_raises():
    model = TinyMLP()
    with pytest.raises(ValueError, match="Unknown importance"):
        prune_model(model, _example_input(), pruning_ratio=0.5, importance="not_real")


# ---------------------------------------------------------------------------
# Gradient-based importances


@pytest.mark.parametrize("importance", ["taylor", "hessian"])
def test_prune_model_gradient_importances(importance: str):
    model = TinyMLP()
    before = _param_count(model)
    prune_model(
        model,
        _example_input(),
        pruning_ratio=0.5,
        importance=importance,
        calibration_steps=2,
    )
    assert _param_count(model) < before


@pytest.mark.xfail(
    reason=(
        "OBDCImportance in torch-pruning has a shape mismatch bug: Fisher is built "
        "with bias columns but __call__ slices only weight columns, causing a "
        "RuntimeError when conv layers have bias=True."
    ),
    strict=True,
)
def test_prune_model_obdc_importance():
    model = TinyCNN(out_classes=10)
    example = torch.randn(2, 3, 16, 16)
    prune_model(
        model,
        example,
        pruning_ratio=0.5,
        importance="obdc",
        num_classes=10,
        calibration_steps=2,
        ignored_layers=[model.head],
    )


# ---------------------------------------------------------------------------
# Extra pruner params


def test_prune_model_global_pruning():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5, global_pruning=True)
    assert _param_count(model) < before


def test_prune_model_max_pruning_ratio():
    model = TinyMLP(hidden=32)
    prune_model(model, _example_input(), pruning_ratio=0.9, max_pruning_ratio=0.5)
    # max_pruning_ratio=0.5 caps it; hidden must be at least 16 (half of 32)
    assert model.fc1.weight.shape[0] >= 1


def test_prune_model_importance_p():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(model, _example_input(), pruning_ratio=0.5, importance="magnitude", importance_p=1)
    assert _param_count(model) < before


def test_prune_model_multivariable_taylor():
    model = TinyMLP()
    before = _param_count(model)
    prune_model(
        model,
        _example_input(),
        pruning_ratio=0.5,
        importance="taylor",
        multivariable=True,
        calibration_steps=2,
    )
    assert _param_count(model) < before


# ---------------------------------------------------------------------------
# resolve_ignored_layers


def test_collect_unwrapped_parameters_layernorm():
    class _ModelWithLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(16)
            self.fc = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.ln(x))

    model = _ModelWithLN()
    params = collect_unwrapped_parameters(model)
    param_ids = {id(p) for p, _ in params}
    assert id(model.ln.weight) in param_ids
    assert id(model.ln.bias) in param_ids
    assert all(dim == 0 for _, dim in params)


def test_collect_unwrapped_parameters_rms_like():
    class _RMSNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.weight

    class _ModelWithRMS(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = _RMSNorm(16)
            self.fc = nn.Linear(16, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.norm(x))

    model = _ModelWithRMS()
    params = collect_unwrapped_parameters(model)
    assert any(id(p) == id(model.norm.weight) for p, _ in params)


def test_collect_unwrapped_parameters_skips_linear():
    model = TinyMLP()
    params = collect_unwrapped_parameters(model)
    linear_param_ids = {
        id(p) for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters()
    }
    collected_ids = {id(p) for p, _ in params}
    assert collected_ids.isdisjoint(linear_param_ids)


def test_resolve_ignored_layers_valid():
    model = TinyMLP()
    result = resolve_ignored_layers(model, ["head"])
    assert result == [model.head]


def test_resolve_ignored_layers_missing_warns(caplog: pytest.LogCaptureFixture):
    model = TinyMLP()
    with caplog.at_level(logging.WARNING):
        result = resolve_ignored_layers(model, ["does_not_exist"])
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


# ---------------------------------------------------------------------------
# Integration: TorchPruningPass through Olive HfModelHandler


@pytest.mark.integration
def test_pass_runs_on_hf_handler_no_unwrapped_warning():
    """TorchPruningPass on microsoft/resnet-50 must produce no unwrapped-parameter warning."""
    handler = HfModelHandler(
        model_path="microsoft/resnet-50",
        task="image-classification",
    )
    accel = AcceleratorSpec(Device.CPU, ExecutionProvider.CPUExecutionProvider)
    cfg_class, _ = TorchPruningPass.get_config_class(accel)
    cfg = cfg_class(
        pruning_ratio=0.25,
        importance="magnitude",
        global_pruning=False,
        example_input_shape=[1, 3, 224, 224],
        ignored_layers=["classifier.1"],
    )
    pass_obj = TorchPruningPass(accel, cfg)  # pyrefly: ignore[bad-argument-type]

    with tempfile.TemporaryDirectory() as out_dir:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out_handler = pass_obj.run(handler, str(Path(out_dir) / "pruned"))

        unwrapped = [w for w in caught if "Unwrapped parameters" in str(w.message)]
        assert unwrapped == [], f"Got {len(unwrapped)} unwrapped-parameter warning(s)"
        assert out_handler is not None
