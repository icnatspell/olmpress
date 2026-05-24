"""Utility functions for structured pruning via Torch-Pruning."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

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


def collect_unwrapped_parameters(
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


def resolve_ignored_layers(model: nn.Module, names: list[str]) -> list[nn.Module]:
    """Convert dot-path module names to nn.Module objects."""
    named = dict(model.named_modules())
    result = []
    for name in names:
        if name in named:
            result.append(named[name])
        else:
            logger.warning("ignored_layer %r not found in model - skipping", name)
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
        "unwrapped_parameters": collect_unwrapped_parameters(model),
    }
    if round_to is not None:
        kwargs["round_to"] = round_to
    if output_transform is not None:
        kwargs["output_transform"] = output_transform

    pruner = tp.pruner.BasePruner(model, example_inputs, **kwargs)

    is_gradient_based = importance in _GRADIENT_BASED
    for _ in range(iterative_steps):
        if is_gradient_based and calibration_steps > 0:
            _accumulate_gradients(model, example_inputs, imp, pruner, calibration_steps)
        pruner.step()

    return model
