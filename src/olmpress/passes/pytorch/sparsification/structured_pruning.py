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

# Importances that require gradient accumulation before pruner.step().
_GRADIENT_BASED = frozenset({"taylor", "hessian", "obdc"})


def _build_importance(  # noqa: PLR0911
    name: str,
    *,
    p: int = 2,
    group_reduction: str = "mean",
    multivariable: bool = False,
    num_classes: int = 100,
) -> Any:  # noqa: ANN401
    import torch_pruning as tp  # noqa: PLC0415

    if name == "magnitude":
        return tp.importance.MagnitudeImportance(p=p, group_reduction=group_reduction)
    if name == "group_magnitude":
        return tp.importance.GroupMagnitudeImportance(p=p, group_reduction=group_reduction)
    if name == "random":
        return tp.importance.RandomImportance()
    if name == "bn_scale":
        return tp.importance.BNScaleImportance(group_reduction=group_reduction)
    if name == "lamp":
        return tp.importance.LAMPImportance(p=p)
    if name == "fpgm":
        return tp.importance.FPGMImportance(p=p, group_reduction=group_reduction)
    if name == "taylor":
        return tp.importance.TaylorImportance(
            group_reduction=group_reduction, multivariable=multivariable
        )
    if name == "hessian":
        return tp.importance.HessianImportance(group_reduction=group_reduction)
    if name == "obdc":
        return tp.importance.OBDCImportance(
            group_reduction=group_reduction, num_classes=num_classes
        )
    valid = [
        "magnitude",
        "group_magnitude",
        "random",
        "bn_scale",
        "lamp",
        "fpgm",
        "taylor",
        "hessian",
        "obdc",
    ]
    msg = f"Unknown importance criterion: {name!r}. Choose from {valid}."
    raise ValueError(msg)


def _obdc_prepare_model(importance: Any, pruner: Any) -> None:  # noqa: ANN401
    """Register OBDC forward/backward hooks for Fisher accumulation.

    Reimplements OBDCImportance._prepare_model without the
    _downstream_node_as_root_if_attention call, which returns None for all
    groups in models without attention layers and causes a TypeError crash.
    """
    from torch_pruning.pruner import function  # noqa: PLC0415

    for group in pruner.DG.get_all_groups(
        ignored_layers=pruner.ignored_layers,
        root_module_types=pruner.root_module_types,
    ):
        for dep, _idxs in group:
            layer = dep.target.module
            if (
                isinstance(layer, tuple(importance.target_types))
                and dep.handler
                in [
                    function.prune_conv_out_channels,
                    function.prune_linear_out_channels,
                ]
                and layer not in importance.modules
            ):
                importance.modules.append(layer)
                layer.register_forward_pre_hook(importance._save_input)  # noqa: SLF001
                layer.register_backward_hook(importance._save_grad_output)  # noqa: SLF001


def _accumulate_gradients(
    model: nn.Module,
    example_inputs: torch.Tensor,
    importance: Any,  # noqa: ANN401
    pruner: Any,  # noqa: ANN401
    steps: int,
) -> None:
    """Run synthetic forward/backward passes to seed gradient-based importances."""
    is_obdc = type(importance).__name__ == "OBDCImportance"
    if is_obdc:
        _obdc_prepare_model(importance, pruner)

    training = model.training
    model.train()
    try:
        for _ in range(steps):
            model.zero_grad()
            out = (
                model(**example_inputs)
                if isinstance(example_inputs, dict)
                else model(example_inputs)
            )
            if isinstance(out, dict):
                loss = sum(v.float().sum() for v in out.values() if isinstance(v, torch.Tensor))
            elif isinstance(out, (tuple, list)):
                loss = out[0].float().sum()
            else:
                loss = out.float().sum()
            loss.backward()
            if is_obdc:
                importance.step()
    finally:
        model.train(training)


def _collect_unwrapped_parameters(
    model: nn.Module,
) -> list[tuple[torch.nn.Parameter, int]]:
    """Return (param, 0) for LayerNorm and RMSNorm-like 1-D scale parameters.

    Custom norm modules (e.g. LlamaRMSNorm) sit outside torch-pruning's normal
    dependency graph. Passing them explicitly as unwrapped_parameters suppresses
    the UserWarning and ensures they are pruned along the correct dimension (0).
    """
    seen: set[int] = set()
    result: list[tuple[torch.nn.Parameter, int]] = []
    _skip = (
        nn.Linear,
        nn.Conv2d,
        nn.Conv1d,
        nn.Embedding,
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm2d,
    )
    for module in model.modules():
        is_layernorm = isinstance(module, nn.LayerNorm)
        is_rms_like = (
            not isinstance(module, _skip)
            and hasattr(module, "weight")
            and isinstance(module.weight, nn.Parameter)
            and module.weight.dim() == 1
        )
        if is_layernorm or is_rms_like:
            for attr in ("weight", "bias"):
                p = getattr(module, attr, None)
                if isinstance(p, nn.Parameter) and id(p) not in seen:
                    seen.add(id(p))
                    result.append((p, 0))
    return result


def prune_model(  # noqa: PLR0913
    model: nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
    *,
    importance: str = "magnitude",
    iterative_steps: int = 1,
    ignored_layers: list[nn.Module] | None = None,
    round_to: int | None = None,
    global_pruning: bool = False,
    max_pruning_ratio: float = 1.0,
    isomorphic: bool = False,
    importance_p: int = 2,
    group_reduction: str = "mean",
    multivariable: bool = False,
    num_classes: int = 100,
    calibration_steps: int = 10,
    output_transform: Any = None,  # noqa: ANN401
) -> nn.Module:
    """Apply structured channel pruning to *model* in-place and return it."""
    import torch_pruning as tp  # noqa: PLC0415

    imp = _build_importance(
        importance,
        p=importance_p,
        group_reduction=group_reduction,
        multivariable=multivariable,
        num_classes=num_classes,
    )
    kwargs: dict[str, Any] = {
        "importance": imp,
        "pruning_ratio": pruning_ratio,
        "ignored_layers": ignored_layers or [],
        "iterative_steps": iterative_steps,
        "global_pruning": global_pruning,
        "max_pruning_ratio": max_pruning_ratio,
        "isomorphic": isomorphic,
        "unwrapped_parameters": _collect_unwrapped_parameters(model),
    }
    if round_to is not None:
        kwargs["round_to"] = round_to
    if output_transform is not None:
        kwargs["output_transform"] = output_transform

    pruner = tp.pruner.MagnitudePruner(model, example_inputs, **kwargs)

    is_gradient_based = importance in _GRADIENT_BASED
    for _ in range(iterative_steps):
        if is_gradient_based and calibration_steps > 0:
            _accumulate_gradients(model, example_inputs, imp, pruner, calibration_steps)
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

    Gradient-based importances (taylor, hessian, obdc) require backward passes
    before each pruning step. The pass runs ``calibration_steps`` synthetic
    forward/backward passes using the example inputs as a proxy.
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
                description=(
                    "Importance criterion. Static: 'magnitude', 'group_magnitude', "
                    "'random', 'bn_scale', 'lamp', 'fpgm'. "
                    "Gradient-based: 'taylor', 'hessian', 'obdc'."
                ),
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
                description="Round pruned channel counts to this multiple (e.g. 8).",
            ),
            "global_pruning": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable global pruning across all layers.",
            ),
            "max_pruning_ratio": PassConfigParam(
                type_=float,
                default_value=1.0,
                description="Maximum per-layer pruning ratio; prevents over-pruning.",
            ),
            "isomorphic": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Enable isomorphic pruning (requires global_pruning=True).",
            ),
            "importance_p": PassConfigParam(
                type_=int,
                default_value=2,
                description="L-p norm order for magnitude/LAMP/FPGM importances.",
            ),
            "group_reduction": PassConfigParam(
                type_=str,
                default_value="mean",
                description="How to reduce group importance scores: 'mean', 'sum', 'max', 'prod'.",
            ),
            "multivariable": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Use multivariable Taylor expansion (Taylor/GroupTaylor only).",
            ),
            "num_classes": PassConfigParam(
                type_=int,
                default_value=100,
                description="Number of output classes for OBDC importance.",
            ),
            "calibration_steps": PassConfigParam(
                type_=int,
                default_value=10,
                description=(
                    "Synthetic forward/backward passes to accumulate gradients for "
                    "gradient-based importances (taylor, hessian, obdc). Ignored otherwise."
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
        cfg = pt_model.config  # type: ignore[union-attr]
        if hasattr(cfg, "vocab_size"):
            example_inputs: Any = torch.randint(0, cfg.vocab_size, list(config.example_input_shape))
        elif hasattr(cfg, "num_channels"):
            # Vision model: infer shape from config; pass as dict so torch-pruning
            # calls model(**inputs) rather than unpacking the tensor with *.
            raw = getattr(cfg, "image_size", 224)
            img = raw if isinstance(raw, int) else raw[0]
            example_inputs = {"pixel_values": torch.randn(1, cfg.num_channels, img, img)}
        else:
            example_inputs = {"pixel_values": torch.randn(list(config.example_input_shape))}

        ignored = _resolve_ignored_layers(pt_model, config.ignored_layers or [])
        # HF models return ModelOutput tuples; output_transform extracts the logits
        # tensor so torch-pruning can trace the dependency graph correctly.
        prune_model(
            pt_model,
            example_inputs=example_inputs,
            pruning_ratio=config.pruning_ratio,
            importance=config.importance,
            iterative_steps=config.iterative_steps,
            ignored_layers=ignored,
            round_to=config.round_to,
            global_pruning=config.global_pruning,
            max_pruning_ratio=config.max_pruning_ratio,
            isomorphic=config.isomorphic,
            importance_p=config.importance_p,
            group_reduction=config.group_reduction,
            multivariable=config.multivariable,
            num_classes=config.num_classes,
            calibration_steps=config.calibration_steps,
            output_transform=lambda out: (out.logits if hasattr(out, "logits") else out[0]).sum(),
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
            global_pruning=config.global_pruning,
            max_pruning_ratio=config.max_pruning_ratio,
            isomorphic=config.isomorphic,
            importance_p=config.importance_p,
            group_reduction=config.group_reduction,
            multivariable=config.multivariable,
            num_classes=config.num_classes,
            calibration_steps=config.calibration_steps,
        )

        out_path = Path(output_model_path).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.trace(pt_model, example_inputs)
        torch.jit.save(scripted, str(out_path))
        return PyTorchModelHandler(
            model_path=str(out_path),
            model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT,
        )
