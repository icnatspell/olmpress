"""TorchPruningPass: Olive pass for structured channel pruning via Torch-Pruning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

from olmpress.passes.pytorch.structured_pruning.utils import prune_model, resolve_ignored_layers

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec


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

        ignored = resolve_ignored_layers(pt_model, config.ignored_layers or [])
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

        ignored = resolve_ignored_layers(pt_model, config.ignored_layers or [])
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
