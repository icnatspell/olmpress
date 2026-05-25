"""FineTunePass: Olive pass that delegates fine-tuning to a user-supplied function."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from olive.model import PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import (
    ParamCategory,
    PassConfigParam,
    get_user_script_data_config,
)

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec


class FineTunePass(Pass):
    """Fine-tuning pass that delegates the training loop to a user-supplied function.

    The user provides a Python script with a function of the form::

        def finetune(model: torch.nn.Module, config: dict) -> torch.nn.Module:
            ...

    ``model`` is the loaded PyTorch model (e.g. a TorchScript ``ScriptModule``).
    ``config`` is the ``finetune_config`` dict from the workflow YAML, passed through
    verbatim so the user controls all hyperparameters.  The function must return the
    fine-tuned model in the same scriptable form it received.

    The pass then saves the returned model and hands it to the next Olive pass.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(required=True),
            "finetune_fn": PassConfigParam(
                type_=Callable | str,
                required=True,
                category=ParamCategory.OBJECT,
                description=(
                    "Fine-tuning function, or its name as a string (resolved from user_script). "
                    "Signature: finetune(model, config) -> model."
                ),
            ),
            "finetune_config": PassConfigParam(
                type_=dict,
                required=False,
                default_value=None,
                description=(
                    "Arbitrary dict passed verbatim to finetune_fn as its second argument. "
                    "Use this to forward hyperparameters (lr, epochs, dataset paths, etc.)."
                ),
            ),
        }

    def _run_for_config(
        self,
        model: Any,
        config: Any,
        output_model_path: str,
    ) -> PyTorchModelHandler:
        from olive.constants import ModelFileFormat

        if not isinstance(model, PyTorchModelHandler):
            msg = f"FineTunePass only supports PyTorchModelHandler, got {type(model).__name__}"
            raise TypeError(msg)

        fn = self._user_module_loader.load_object(config.finetune_fn)
        pt_model = model.load_model()
        fine_tuned = fn(pt_model, config.finetune_config or {})

        out_path = Path(output_model_path).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(fine_tuned, str(out_path))

        return PyTorchModelHandler(
            model_path=str(out_path),
            model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT,
        )
