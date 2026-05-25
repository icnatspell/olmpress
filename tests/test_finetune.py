"""Tests for FineTunePass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from olive.constants import ModelFileFormat
from olive.hardware.accelerator import AcceleratorSpec, Device, ExecutionProvider
from olive.model import PyTorchModelHandler
from olive.passes import Pass
from torch import nn

from chisel.passes.pytorch.finetune import FineTunePass

if TYPE_CHECKING:
    from pathlib import Path


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_scripted_handler(tmp_path: Path) -> PyTorchModelHandler:
    scripted = torch.jit.script(_TinyModel())
    model_path = tmp_path / "input.pt"
    torch.jit.save(scripted, str(model_path))
    return PyTorchModelHandler(
        model_path=str(model_path),
        model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT,
    )


def _make_pass(user_script: Path, fn_name: str) -> FineTunePass:
    accel = AcceleratorSpec(Device.CPU, ExecutionProvider.CPUExecutionProvider)
    cfg_class, _ = FineTunePass.get_config_class(accel)
    cfg = cfg_class(user_script=str(user_script), finetune_fn=fn_name, finetune_config={})
    return FineTunePass(accel, cfg)  # pyrefly: ignore[bad-argument-type]


def test_finetune_pass_is_registered():
    assert "finetunepass" in Pass.registry


def test_finetune_pass_calls_user_function_and_returns_handler(tmp_path: Path):
    script = tmp_path / "user_script.py"
    script.write_text(
        "def my_finetune(model, config):\n"
        "    return model  # already a ScriptModule — pass through\n"
    )

    handler = _make_scripted_handler(tmp_path)
    pass_obj = _make_pass(script, "my_finetune")

    out_handler = pass_obj.run(handler, str(tmp_path / "out"))

    assert isinstance(out_handler, PyTorchModelHandler)
    loaded = out_handler.load_model()
    assert isinstance(loaded, torch.jit.ScriptModule)
    out = loaded(torch.randn(1, 4))
    assert out.shape == (1, 2)


def test_finetune_pass_passes_config_to_user_function(tmp_path: Path):
    sentinel_path = tmp_path / "sentinel.txt"
    script = tmp_path / "user_script.py"
    script.write_text(
        "from pathlib import Path\n"
        "def my_finetune(model, config):\n"
        f"    Path({str(sentinel_path)!r}).write_text(str(config['marker']))\n"
        "    return model\n"
    )

    handler = _make_scripted_handler(tmp_path)
    accel = AcceleratorSpec(Device.CPU, ExecutionProvider.CPUExecutionProvider)
    cfg_class, _ = FineTunePass.get_config_class(accel)
    cfg = cfg_class(
        user_script=str(script),
        finetune_fn="my_finetune",
        finetune_config={"marker": "hello"},
    )
    pass_obj = FineTunePass(accel, cfg)  # pyrefly: ignore[bad-argument-type]
    pass_obj.run(handler, str(tmp_path / "out"))

    assert sentinel_path.read_text() == "hello"


def test_finetune_pass_rejects_non_pytorch_handler(tmp_path: Path):
    from olive.model import ONNXModelHandler

    script = tmp_path / "user_script.py"
    script.write_text("def f(model, config): return model\n")

    # ONNXModelHandler is not a PyTorchModelHandler; constructor takes a path
    # but FineTunePass should reject before loading.
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"")
    handler = ONNXModelHandler(model_path=str(onnx_path))

    pass_obj = _make_pass(script, "f")
    with pytest.raises(TypeError, match="only supports PyTorchModelHandler"):
        pass_obj.run(handler, str(tmp_path / "out"))


def test_finetune_pass_rejects_non_script_module_return(tmp_path: Path):
    script = tmp_path / "user_script.py"
    script.write_text(
        "import torch\n"
        "def bad_finetune(model, config):\n"
        "    return torch.nn.Linear(4, 2)  # plain nn.Module, not ScriptModule\n"
    )

    handler = _make_scripted_handler(tmp_path)
    pass_obj = _make_pass(script, "bad_finetune")

    with pytest.raises(TypeError, match="ScriptModule"):
        pass_obj.run(handler, str(tmp_path / "out"))
