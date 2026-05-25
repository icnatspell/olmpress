"""FineTunePass: Olive pass that delegates fine-tuning to a user-supplied function."""

from __future__ import annotations

import atexit
import os
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

        def finetune(model: torch.jit.ScriptModule, config: dict) -> torch.jit.ScriptModule:
            ...

    ``model`` is the loaded TorchScript module produced by an upstream pass (e.g.
    ``TorchPruningPass``).  ``config`` is the ``finetune_config`` dict from the
    workflow YAML, passed through verbatim so the user controls all
    hyperparameters.  The function must return a ``torch.jit.ScriptModule`` so the
    pass can save it for the next stage of the pipeline.

    The pass registers ``atexit(os._exit(0))`` before invoking the user function.
    This bypasses Python's normal shutdown cleanup, which can hang indefinitely
    on open HTTP connections from HuggingFace streaming datasets — a common
    pattern inside user fine-tuning scripts.
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

        # See class docstring: prevents shutdown hangs on HuggingFace streaming datasets.
        atexit.register(os._exit, 0)

        fn = self._user_module_loader.load_object(config.finetune_fn)
        pt_model = model.load_model()
        fine_tuned = fn(pt_model, config.finetune_config or {})

        if not isinstance(fine_tuned, torch.jit.ScriptModule):
            msg = (
                f"finetune_fn must return a torch.jit.ScriptModule, got "
                f"{type(fine_tuned).__name__}. The input model is a TorchScript module — "
                "keep it scripted throughout training (e.g., don't unwrap it)."
            )
            raise TypeError(msg)

        out_path = Path(output_model_path).with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(fine_tuned, str(out_path))

        return PyTorchModelHandler(
            model_path=str(out_path),
            model_file_format=ModelFileFormat.PYTORCH_TORCH_SCRIPT,
        )
