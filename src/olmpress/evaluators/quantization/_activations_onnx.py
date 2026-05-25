"""Intermediate activation capture for ONNX models via ONNX Runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import onnx
import onnxruntime as ort

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import numpy as np


def list_intermediate_tensors(model: onnx.ModelProto) -> list[str]:
    """Return every node-output tensor name, excluding initializers and graph outputs."""
    initializer_names = {init.name for init in model.graph.initializer}
    existing_outputs = {out.name for out in model.graph.output}
    return [
        output_name
        for node in model.graph.node
        for output_name in node.output
        if output_name
        and output_name not in initializer_names
        and output_name not in existing_outputs
    ]


def _expose_outputs(model: onnx.ModelProto, tensor_names: Iterable[str]) -> onnx.ModelProto:
    requested = list(dict.fromkeys(tensor_names))
    existing_outputs = {out.name for out in model.graph.output}

    clone = onnx.ModelProto()
    clone.ParseFromString(model.SerializeToString())

    for name in requested:
        if name in existing_outputs:
            continue
        value_info = onnx.ValueInfoProto()
        value_info.name = name
        clone.graph.output.append(value_info)
    return clone


def capture_onnx(
    model: onnx.ModelProto,
    tensor_names: Iterable[str],
    inputs: Mapping[str, np.ndarray],
    *,
    providers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run ``model`` and return ``{tensor_name: ndarray}`` for the requested tensors."""
    requested = list(dict.fromkeys(tensor_names))
    producible = set(list_intermediate_tensors(model)) | {out.name for out in model.graph.output}
    missing = [name for name in requested if name not in producible]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""  # noqa: PLR2004
        msg = f"capture_onnx: unknown tensor name(s): {preview}{more}"
        raise KeyError(msg)

    augmented = _expose_outputs(model, requested)
    session_providers = providers if providers is not None else ort.get_available_providers()
    session = ort.InferenceSession(
        augmented.SerializeToString(),
        providers=session_providers,
    )

    output_names = [out.name for out in session.get_outputs()]
    results = session.run(output_names, dict(inputs))
    by_name = dict(zip(output_names, results, strict=True))
    return {name: cast("np.ndarray", by_name[name]) for name in requested}
