"""Quantization-degradation evaluator for Olive."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from olive.evaluator.metric_result import MetricResult, SubMetricResult, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.evaluator.registry import Registry
from olive.hardware import Device
from olive.model.handler.onnx import ONNXModelHandler
from olive.model.handler.pytorch import PyTorchModelHandlerBase
from torch import nn

from olmpress.activations import capture as capture_pt
from olmpress.activations_onnx import capture_onnx
from olmpress.mapping import View, build_mapping, select_view
from olmpress.metrics import cosine_similarity, kl_divergence, mse, relative_l2, sqnr

if TYPE_CHECKING:
    from collections.abc import Callable

    from olive.evaluator.metric import Metric
    from olive.model.handler.base import OliveModelHandler


_SUB_TYPE_TABLE: dict[str, tuple[str, str | None]] = {
    "sqnr_mean": ("sqnr", "mean"),
    "sqnr_min": ("sqnr", "min"),
    "sqnr_max": ("sqnr", "max"),
    "sqnr_p50": ("sqnr", "p50"),
    "cosine_mean": ("cosine", "mean"),
    "cosine_min": ("cosine", "min"),
    "mse_mean": ("mse", "mean"),
    "mse_max": ("mse", "max"),
    "relative_l2_mean": ("relative_l2", "mean"),
    "relative_l2_max": ("relative_l2", "max"),
    "kl": ("kl", None),
}


def supported_sub_types() -> tuple[str, ...]:
    """Return the sub-type names this evaluator can produce."""
    return tuple(_SUB_TYPE_TABLE)


def _aggregate(per_layer: dict[str, float], how: str) -> float:
    if not per_layer:
        return float("nan")
    values = np.array(list(per_layer.values()), dtype=np.float64)
    if how == "mean":
        return float(values.mean())
    if how == "min":
        return float(values.min())
    if how == "max":
        return float(values.max())
    if how == "p50":
        return float(np.median(values))
    msg = f"Unknown aggregation: {how!r}"
    raise ValueError(msg)


_PER_LAYER_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "sqnr": sqnr,
    "cosine": cosine_similarity,
    "mse": mse,
    "relative_l2": relative_l2,
}

_HIGHER_IS_BETTER: dict[str, bool] = {
    "sqnr": True,
    "cosine": True,
    "mse": False,
    "relative_l2": False,
    "kl": False,
}


def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def _compute_per_layer(
    metric_kind: str,
    ref_caps: dict[str, torch.Tensor | np.ndarray],
    tgt_caps: dict[str, torch.Tensor | np.ndarray],
    mapping: dict[str, str],
) -> dict[str, float]:
    fn = _PER_LAYER_FNS[metric_kind]
    out: dict[str, float] = {}
    for ref_name, tgt_name in mapping.items():
        if ref_name not in ref_caps or tgt_name not in tgt_caps:
            continue
        ref = _to_tensor(ref_caps[ref_name])
        tgt = _to_tensor(tgt_caps[tgt_name])
        if ref.shape != tgt.shape:
            continue
        out[ref_name] = float(fn(ref, tgt).item())
    return out


def _compute_kl(
    ref_caps: dict[str, torch.Tensor | np.ndarray],
    tgt_caps: dict[str, torch.Tensor | np.ndarray],
    logits_key_ref: str,
    logits_key_tgt: str,
    *,
    temperature: float,
) -> float:
    ref = _to_tensor(ref_caps[logits_key_ref])
    tgt = _to_tensor(tgt_caps[logits_key_tgt])
    return float(kl_divergence(ref, tgt, temperature=temperature).item())


@Registry.register("olmpress_degradation")
class DegradationEvaluator(OliveEvaluator):
    """Olive evaluator that measures per-layer quantization error."""

    _evaluator_type: ClassVar[str] = "olmpress_degradation"

    def __init__(  # noqa: PLR0913
        self,
        reference_model: Callable[[], nn.Module | onnx.ModelProto] | dict[str, Any] | None = None,
        inputs: Callable[[], dict[str, Any]] | dict[str, Any] | None = None,
        *,
        view: View = "all",
        temperature: float = 1.0,
        logits_layer: str = "lm_head",
        rename: dict[str, str] | None = None,
        cpu_captures: bool = False,
        cross_framework_precision: str = "fp32",
        cross_framework_cache_dir: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Configure the evaluator from callable loaders or JSON-shaped specs."""
        super().__init__(**kwargs)
        self._reference_spec: dict[str, Any] | None = (
            reference_model if isinstance(reference_model, dict) else None
        )
        self._reference_loader: Callable[[], nn.Module | onnx.ModelProto] | None = (
            _coerce_model_loader(reference_model)
        )
        self._inputs_loader: Callable[[], dict[str, Any]] | None = _coerce_inputs_loader(inputs)
        self._view: View = view
        self._temperature = temperature
        self._logits_layer = logits_layer
        self._rename = rename
        self._cpu_captures = cpu_captures
        self._cross_framework_precision = cross_framework_precision
        self._cross_framework_cache_dir = cross_framework_cache_dir
        self._onnx_reference_cache: onnx.ModelProto | None = None

    def evaluate(  # type: ignore[override]  # pyrefly: ignore [bad-override]
        self,
        model: OliveModelHandler | nn.Module | onnx.ModelProto,
        metrics: list[Metric],
        device: Device = Device.CPU,
        execution_providers: str | list[str] | None = None,
    ) -> MetricResult:
        """Run reference vs target and return the requested sub-metrics."""
        del device
        if self._reference_loader is None or self._inputs_loader is None:
            msg = "DegradationEvaluator requires both reference_model and inputs."
            raise RuntimeError(msg)

        reference = self._reference_loader()
        inputs = self._inputs_loader()
        target = _load_target(model)

        if isinstance(reference, nn.Module) and isinstance(target, nn.Module):
            ref_caps, tgt_caps, mapping = self._run_pytorch(reference, target, inputs)
        elif isinstance(reference, onnx.ModelProto) and isinstance(target, onnx.ModelProto):
            ref_caps, tgt_caps, mapping = self._run_onnx(
                reference, target, inputs, execution_providers
            )
        elif isinstance(reference, nn.Module) and isinstance(target, onnx.ModelProto):
            ref_caps, tgt_caps, mapping = self._run_cross_framework(
                target, inputs, execution_providers
            )
        else:
            msg = (
                f"DegradationEvaluator: reference/target type mismatch — "
                f"reference={type(reference).__name__}, target={type(target).__name__}."
            )
            raise TypeError(msg)

        return self._build_result(
            metrics,
            cast("dict[str, torch.Tensor | np.ndarray]", ref_caps),
            cast("dict[str, torch.Tensor | np.ndarray]", tgt_caps),
            mapping,
        )

    def _run_pytorch(
        self,
        reference: nn.Module,
        target: nn.Module,
        inputs: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
        full_mapping = build_mapping(reference, target, rename=self._rename)
        viewed = select_view(full_mapping, reference, self._view)
        # Always capture logits_layer so KL is available regardless of the view.
        ref_names = sorted(set(viewed) | {self._logits_layer})
        tgt_names_for_capture = sorted({full_mapping.get(n, n) for n in ref_names})

        for m in (reference, target):
            m.eval()
        with torch.inference_mode():
            with capture_pt(reference, ref_names, cpu=self._cpu_captures) as rc:
                reference(**inputs)
            ref_caps = dict(rc)
            with capture_pt(target, tgt_names_for_capture, cpu=self._cpu_captures) as tc:
                target(**inputs)
            tgt_caps = dict(tc)
        return ref_caps, tgt_caps, viewed

    def _run_cross_framework(
        self,
        target: onnx.ModelProto,
        inputs: dict[str, Any],
        execution_providers: str | list[str] | None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
        onnx_reference = self._get_onnx_reference()
        # Force CPU: avoids CUDA-specific dtype requirements on GQA ops.
        cpu = ["CPUExecutionProvider"]
        np_inputs = {k: np.asarray(v) for k, v in inputs.items()}
        # Supplement inputs separately: INT4 uses float16 KV-cache, fp32 uses float32.
        ref_inputs = _supplement_onnx_inputs(onnx_reference, np_inputs, cpu)
        tgt_inputs = _supplement_onnx_inputs(target, np_inputs, cpu)

        ref_names = _onnx_node_outputs(onnx_reference)
        tgt_names = _onnx_node_outputs(target)
        rename = self._rename or {}
        mapping = {n: rename.get(n, n) for n in ref_names if rename.get(n, n) in tgt_names}
        mapping = _apply_onnx_view(mapping, self._view)
        if self._logits_layer in ref_names and self._logits_layer in tgt_names:
            mapping[self._logits_layer] = rename.get(self._logits_layer, self._logits_layer)

        ref_caps = capture_onnx(onnx_reference, list(mapping.keys()), ref_inputs, providers=cpu)
        tgt_caps = capture_onnx(target, list(mapping.values()), tgt_inputs, providers=cpu)
        return ref_caps, tgt_caps, mapping

    def _get_onnx_reference(self) -> onnx.ModelProto:
        if self._onnx_reference_cache is not None:
            return self._onnx_reference_cache

        if self._reference_spec is None:
            msg = (
                "DegradationEvaluator: cross-framework comparison requires reference_model "
                "to be a dict spec (e.g., {type: HfModel, config: {model_path: ...}})"
            )
            raise RuntimeError(msg)

        spec = self._reference_spec
        model_path = (spec.get("config") or spec).get("model_path")
        if model_path is None:
            msg = (
                "DegradationEvaluator: cross-framework conversion requires "
                "model_path in reference_model"
            )
            raise RuntimeError(msg)

        import tempfile  # noqa: PLC0415

        from olive.hardware.accelerator import (  # noqa: PLC0415
            AcceleratorSpec,
            Device,
            ExecutionProvider,
        )
        from olive.model import ModelConfig  # noqa: PLC0415
        from olive.passes.onnx.model_builder import ModelBuilder  # noqa: PLC0415

        cache_dir = self._cross_framework_cache_dir or tempfile.mkdtemp(prefix="olmpress_ref_")
        input_handler = ModelConfig.model_validate(
            {"type": "HfModel", "config": {"model_path": model_path}}
        ).create_model()
        accel = AcceleratorSpec(Device.CPU, ExecutionProvider.CPUExecutionProvider)
        cfg_class, _ = ModelBuilder.get_config_class(accel)
        # Resolve search-space defaults to concrete values so ModelBuilder doesn't receive
        # Categorical objects when called outside Olive's search engine.
        cfg = cfg_class(
            precision=self._cross_framework_precision,
            int4_block_size=32,
            int4_is_symmetric=True,
            int4_algo_config="default",
        )
        mb = ModelBuilder(accel, cfg)  # pyrefly: ignore[bad-argument-type]
        output_handler = mb.run(input_handler, cache_dir)
        loaded = output_handler.load_model()
        if not isinstance(loaded, onnx.ModelProto):
            msg = f"ModelBuilder returned {type(loaded).__name__}, expected ModelProto"
            raise TypeError(msg)
        self._onnx_reference_cache = loaded
        return loaded

    def _run_onnx(
        self,
        reference: onnx.ModelProto,
        target: onnx.ModelProto,
        inputs: dict[str, Any],
        execution_providers: str | list[str] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
        ref_tensor_names = _onnx_node_outputs(reference)
        tgt_tensor_names = _onnx_node_outputs(target)
        rename = self._rename or {}
        mapping = {
            n: rename.get(n, n) for n in ref_tensor_names if rename.get(n, n) in tgt_tensor_names
        }
        mapping = _apply_onnx_view(mapping, self._view)

        if self._logits_layer in ref_tensor_names and self._logits_layer in tgt_tensor_names:
            mapping[self._logits_layer] = rename.get(self._logits_layer, self._logits_layer)

        np_inputs = {k: np.asarray(v) for k, v in inputs.items()}
        providers = (
            [execution_providers] if isinstance(execution_providers, str) else execution_providers
        )
        ref_caps = capture_onnx(reference, list(mapping.keys()), np_inputs, providers=providers)
        tgt_caps = capture_onnx(target, list(mapping.values()), np_inputs, providers=providers)
        return ref_caps, tgt_caps, mapping

    def _build_result(
        self,
        metrics: list[Metric],
        ref_caps: dict[str, torch.Tensor | np.ndarray],
        tgt_caps: dict[str, torch.Tensor | np.ndarray],
        mapping: dict[str, str],
    ) -> MetricResult:
        root: dict[str, SubMetricResult] = {}
        for metric in metrics:
            for sub in metric.sub_types:
                sub_name = sub.name
                if sub_name not in _SUB_TYPE_TABLE:
                    msg = (
                        f"DegradationEvaluator: unknown sub_type {sub_name!r}. "
                        f"Supported: {sorted(_SUB_TYPE_TABLE)}"
                    )
                    raise ValueError(msg)
                kind, agg = _SUB_TYPE_TABLE[sub_name]
                if kind == "kl":
                    value = _compute_kl(
                        ref_caps,
                        tgt_caps,
                        self._logits_layer,
                        mapping.get(self._logits_layer, self._logits_layer),
                        temperature=self._temperature,
                    )
                else:
                    per_layer = _compute_per_layer(kind, ref_caps, tgt_caps, mapping)
                    value = _aggregate(per_layer, agg) if agg else float("nan")
                key = joint_metric_key(metric.name, sub_name)
                root[key] = SubMetricResult(
                    value=value,
                    priority=sub.priority,
                    higher_is_better=_HIGHER_IS_BETTER[kind],
                )
        return MetricResult(root=root)


def _apply_onnx_view(mapping: dict[str, str], view: View) -> dict[str, str]:
    """Filter an ONNX tensor-name mapping to the subset implied by *view*."""
    if view == "all":
        return mapping
    if view == "logits":
        return {}
    if view in ("linears", "blocks"):
        # Keep only MatMul/Gemm node outputs — these correspond to linear layer activations.
        return {ref: tgt for ref, tgt in mapping.items() if "/MatMul/" in ref or "/Gemm/" in ref}
    msg = f"Unknown view: {view!r}"
    raise ValueError(msg)


_ORT_DTYPE_TO_NUMPY: dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(bool)": np.bool_,
}


def _supplement_onnx_inputs(
    model: onnx.ModelProto,
    provided: dict[str, np.ndarray],
    providers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Add any inputs the ONNX model needs that aren't in *provided* (e.g. KV-cache)."""
    sess = ort.InferenceSession(
        model.SerializeToString(),
        providers=providers or ort.get_available_providers(),
    )
    input_ids = provided.get("input_ids")
    batch_size = int(input_ids.shape[0]) if input_ids is not None else 1
    seq_len = int(input_ids.shape[1]) if input_ids is not None else 1
    dim_map: dict[str, int] = {
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "past_sequence_length": 0,
        "total_sequence_length": seq_len,
    }
    result = dict(provided)
    for inp in sess.get_inputs():
        if inp.name in result:
            continue
        shape = [dim if isinstance(dim, int) else dim_map.get(dim, 0) for dim in inp.shape]
        np_dtype = _ORT_DTYPE_TO_NUMPY.get(inp.type, np.float32)
        if inp.name == "attention_mask":
            result[inp.name] = np.ones(shape, dtype=np_dtype)
        else:
            result[inp.name] = np.zeros(shape, dtype=np_dtype)
    return result


def _load_target(
    model: OliveModelHandler | nn.Module | onnx.ModelProto,
) -> nn.Module | onnx.ModelProto:
    if isinstance(model, (nn.Module, onnx.ModelProto)):
        return model
    if isinstance(model, PyTorchModelHandlerBase):
        loaded = model.load_model()
        if not isinstance(loaded, nn.Module):
            msg = (
                f"PyTorchModelHandler.load_model returned {type(loaded).__name__}, "
                "expected nn.Module."
            )
            raise TypeError(msg)
        return loaded
    if isinstance(model, ONNXModelHandler):
        loaded = model.load_model()
        if not isinstance(loaded, onnx.ModelProto):
            msg = (
                f"ONNXModelHandler.load_model returned {type(loaded).__name__}, "
                "expected onnx.ModelProto."
            )
            raise TypeError(msg)
        return loaded
    msg = f"Unsupported model type: {type(model).__name__}"
    raise TypeError(msg)


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "long": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "bool": torch.bool,
}

_NUMPY_DTYPES: dict[str, type] = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "int8": np.int8,
    "bool": np.bool_,
}

_TORCH_INT_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.int8, torch.int32, torch.int64, torch.bool}
)


def make_inputs_loader(spec: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    """Build an inputs loader from a JSON-friendly ``{name: {shape, dtype, ...}}`` spec."""

    def loader() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, entry in spec.items():
            shape = list(entry["shape"])
            dtype_name = str(entry.get("dtype", "float32"))
            framework = entry.get("framework", "torch")
            seed = entry.get("seed", 0)

            if framework == "numpy":
                np_dtype = _NUMPY_DTYPES.get(dtype_name, np.float32)
                rng = np.random.default_rng(seed)
                if "value" in entry:
                    out[name] = np.full(shape, entry["value"], dtype=np_dtype)
                elif np.issubdtype(np_dtype, np.integer):
                    low, high = entry.get("low", 0), entry.get("high", 100)
                    out[name] = rng.integers(low, high, size=shape, dtype=np_dtype)
                else:
                    out[name] = rng.standard_normal(size=shape).astype(np_dtype)
            else:
                torch_dtype = _TORCH_DTYPES.get(dtype_name, torch.float32)
                generator = torch.Generator().manual_seed(seed)
                if "value" in entry:
                    out[name] = torch.full(shape, entry["value"], dtype=torch_dtype)
                elif torch_dtype in _TORCH_INT_DTYPES:
                    low, high = entry.get("low", 0), entry.get("high", 100)
                    out[name] = torch.randint(
                        low, high, size=tuple(shape), dtype=torch_dtype, generator=generator
                    )
                else:
                    out[name] = torch.randn(shape, dtype=torch_dtype, generator=generator)
        return out

    return loader


def _coerce_model_loader(
    spec: Callable[[], nn.Module | onnx.ModelProto] | dict[str, Any] | None,
) -> Callable[[], nn.Module | onnx.ModelProto] | None:
    if spec is None or callable(spec):
        return spec  # type: ignore[return-value]
    if isinstance(spec, dict):
        from olive.model import ModelConfig  # noqa: PLC0415

        config = ModelConfig.model_validate(spec)

        def load() -> nn.Module | onnx.ModelProto:
            handler = config.create_model()
            loaded = handler.load_model()
            if not isinstance(loaded, (nn.Module, onnx.ModelProto)):
                msg = (
                    f"reference_model loaded {type(loaded).__name__}, expected nn.Module or "
                    "onnx.ModelProto."
                )
                raise TypeError(msg)
            return loaded

        return load
    msg = f"Unsupported reference_model spec: {type(spec).__name__}"
    raise TypeError(msg)


def _coerce_inputs_loader(
    spec: Callable[[], dict[str, Any]] | dict[str, Any] | None,
) -> Callable[[], dict[str, Any]] | None:
    if spec is None or callable(spec):
        return spec  # type: ignore[return-value]
    if isinstance(spec, dict):
        return make_inputs_loader(spec)
    msg = f"Unsupported inputs spec: {type(spec).__name__}"
    raise TypeError(msg)


def _onnx_node_outputs(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for node in model.graph.node:
        for out in node.output:
            if out:
                names.add(out)
    for out in model.graph.output:
        names.add(out.name)
    return names
