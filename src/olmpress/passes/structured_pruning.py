"""Structured pruning pass using Torch-Pruning (https://github.com/VainF/Torch-Pruning)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from torch import nn

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec

logger = logging.getLogger(__name__)


def _build_importance(name: str) -> object:
    import torch_pruning as tp  # noqa: PLC0415

    registry = {
        "magnitude": tp.importance.MagnitudeImportance,
        "random": tp.importance.RandomImportance,
        "bn_scale": tp.importance.BNScaleImportance,
    }
    if name not in registry:
        msg = f"Unknown importance criterion: {name!r}. Choose from {list(registry)}."
        raise ValueError(msg)
    return registry[name]()


def prune_model(  # noqa: PLR0913
    model: nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
    importance: str = "magnitude",
    iterative_steps: int = 1,
    ignored_layers: list[nn.Module] | None = None,
    round_to: int | None = None,
) -> nn.Module:
    """Apply structured channel pruning to *model* in-place.

    Returns the same (now smaller) model for convenience.
    """
    import torch_pruning as tp  # noqa: PLC0415

    imp = _build_importance(importance)
    kwargs: dict[str, Any] = {
        "importance": imp,
        "pruning_ratio": pruning_ratio,
        "ignored_layers": ignored_layers or [],
        "iterative_steps": iterative_steps,
    }
    if round_to is not None:
        kwargs["round_to"] = round_to

    pruner = tp.pruner.MagnitudePruner(model, example_inputs, **kwargs)
    for _ in range(iterative_steps):
        pruner.step()
    return model


def _resolve_ignored_layers(model: nn.Module, names: list[str]) -> list[nn.Module]:
    """Convert dot-path module names to nn.Module objects."""
    named = dict(model.named_modules())
    result = []
    for name in names:
        if name in named:
            result.append(named[name])
        else:
            logger.warning("ignored_layer %r not found in model - skipping", name)
    return result


class TorchPruningPass(Pass):
    """Structured pruning pass backed by Torch-Pruning.

    Accepts HfModelHandler or PyTorchModelHandler. The pruned model is saved
    and returned in the same handler type.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:  # noqa: ARG003
        return {
            "pruning_ratio": PassConfigParam(
                type_=float,
                default_value=0.5,
                description="Fraction of channels to prune per layer (0.0-1.0).",
            ),
            "importance": PassConfigParam(
                type_=str,
                default_value="magnitude",
                description="Importance criterion: 'magnitude', 'random', or 'bn_scale'.",
            ),
            "iterative_steps": PassConfigParam(
                type_=int,
                default_value=1,
                description="Number of iterative pruning steps.",
            ),
            "ignored_layers": PassConfigParam(
                type_=list,
                default_value=None,
                description="Dot-path module names to exclude from pruning (e.g. ['lm_head']).",
            ),
            "round_to": PassConfigParam(
                type_=int,
                default_value=None,
                description=(
                    "Round pruned channel counts to this multiple (e.g. 8 for hardware efficiency)."
                ),
            ),
            "example_input_shape": PassConfigParam(
                type_=list,
                default_value=[1, 8],
                description=(
                    "Shape of the dummy input_ids tensor used for dependency-graph tracing "
                    "(HfModelHandler only; ignored for PyTorchModelHandler)."
                ),
            ),
        }

    def _run_for_config(
        self,
        model: Any,  # noqa: ANN401
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Any:  # noqa: ANN401
        if isinstance(model, HfModelHandler):
            return self._run_hf(model, config, output_model_path)
        if isinstance(model, PyTorchModelHandler):
            return self._run_pytorch(model, config, output_model_path)
        msg = f"TorchPruningPass: unsupported model type {type(model).__name__}"
        raise TypeError(msg)

    def _run_hf(
        self,
        handler: HfModelHandler,
        config: Any,  # noqa: ANN401
        output_model_path: str,
    ) -> HfModelHandler:
        pt_model = handler.load_model()
        vocab_size = pt_model.config.vocab_size  # type: ignore[union-attr]
        example_inputs = torch.randint(0, vocab_size, list(config.example_input_shape))

        ignored = _resolve_ignored_layers(pt_model, config.ignored_layers or [])
        prune_model(
            pt_model,
            example_inputs=example_inputs,
            pruning_ratio=config.pruning_ratio,
            importance=config.importance,
            iterative_steps=config.iterative_steps,
            ignored_layers=ignored,
            round_to=config.round_to,
        )

        out_dir = Path(output_model_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        pt_model.save_pretrained(str(out_dir))  # type: ignore[union-attr]
        return HfModelHandler(model_path=str(out_dir), task=handler.task)

    def _run_pytorch(
        self,
        handler: PyTorchModelHandler,
        config: Any,  # noqa: ANN401
        output_model_path: str,
    ) -> PyTorchModelHandler:
        from olive.constants import ModelFileFormat  # noqa: PLC0415

        pt_model = handler.load_model()
        raw: Any = handler.get_dummy_inputs()
        example_inputs: torch.Tensor = next(iter(raw.values())) if isinstance(raw, dict) else raw

        ignored = _resolve_ignored_layers(pt_model, config.ignored_layers or [])
        prune_model(
            pt_model,
            example_inputs=example_inputs,
            pruning_ratio=config.pruning_ratio,
            importance=config.importance,
            iterative_steps=config.iterative_steps,
            ignored_layers=ignored,
            round_to=config.round_to,
        )

        out_path = Path(output_model_path).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.trace(pt_model, example_inputs)
        torch.jit.save(scripted, str(out_path))
        return PyTorchModelHandler(
            model_path=str(out_path),
            model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT,
        )
