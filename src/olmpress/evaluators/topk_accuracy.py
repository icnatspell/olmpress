"""Top-k accuracy evaluator for ImageNet-pretrained models using HuggingFace datasets.

Uses ``zh-plus/tiny-imagenet`` (200 classes, public, no auth required) as the
evaluation dataset.  Ground-truth labels are WordNet IDs which are mapped to
their ImageNet-1k class indices via the TensorFlow class-index JSON so that
models pretrained on full ImageNet-1k are evaluated correctly.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from olive.evaluator.metric_result import MetricResult, SubMetricResult, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.evaluator.registry import Registry
from olive.hardware import Device
from olive.model.handler.pytorch import PyTorchModelHandlerBase
from torch import nn
from torchvision import transforms

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric
    from olive.model.handler.base import OliveModelHandler

logger = logging.getLogger(__name__)

# Maps ImageNet class index (0-999) → [wnid, human-readable name].
# Same ordering torchvision uses for ResNet/ViT/etc. weights.
_IMAGENET_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)

# Standard ImageNet normalisation applied by all torchvision IMAGENET1K_V1 weights.
_IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _wnid_to_imagenet_idx(timeout: int = 30) -> dict[str, int]:
    """Download and parse the TF imagenet_class_index.json → {wnid: class_index}."""
    with urllib.request.urlopen(_IMAGENET_INDEX_URL, timeout=timeout) as resp:  # noqa: S310
        raw = json.loads(resp.read())
    return {v[0]: int(k) for k, v in raw.items()}


def _build_label_map(wnids: list[str], wnid_to_idx: dict[str, int]) -> list[int | None]:
    """Return a list mapping dataset label index → ImageNet class index (or None)."""
    mapping: list[int | None] = []
    unknown: list[str] = []
    for wnid in wnids:
        idx = wnid_to_idx.get(wnid)
        mapping.append(idx)
        if idx is None:
            unknown.append(wnid)
    if unknown:
        logger.warning("wnids not in ImageNet index (first 5): %s", unknown[:5])
    return mapping


@torch.no_grad()
def _run_topk(
    model: nn.Module,
    dataset: Any,  # noqa: ANN401
    label_map: list[int | None],
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (top1_accuracy, top5_accuracy) over the dataset."""
    model.eval()
    model = model.to(device)

    correct1 = correct5 = total = 0

    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))

        imgs: list[torch.Tensor] = []
        targets: list[int] = []
        for sample in batch:
            img = sample["image"].convert("RGB")
            inet_idx = label_map[sample["label"]]
            if inet_idx is None:
                continue
            imgs.append(_IMAGENET_TRANSFORM(img))
            targets.append(inet_idx)

        if not imgs:
            continue

        x = torch.stack(imgs).to(device)
        gt = torch.tensor(targets, device=device)

        logits = model(x)
        # ScriptModule may return a namedtuple; normalise to plain tensor.
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]

        _, top5_preds = logits.topk(5, dim=1)
        correct1 += int((top5_preds[:, 0] == gt).sum())
        correct5 += int((top5_preds == gt.unsqueeze(1)).any(dim=1).sum())
        total += len(gt)

    if total == 0:
        return float("nan"), float("nan")
    return correct1 / total, correct5 / total


@Registry.register("olmpress_imagenet_accuracy")
class ImageNetAccuracyEvaluator(OliveEvaluator):
    """Olive evaluator measuring top-1 / top-5 accuracy on a Tiny-ImageNet subset.

    Supported sub-type names (used in the YAML ``sub_types`` list):

    * ``top1`` — top-1 accuracy (higher is better)
    * ``top5`` — top-5 accuracy (higher is better)
    """

    _evaluator_type: ClassVar[str] = "olmpress_imagenet_accuracy"

    def __init__(
        self,
        *,
        dataset: str = "zh-plus/tiny-imagenet",
        split: str = "valid",
        num_samples: int = 1000,
        batch_size: int = 32,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Configure dataset name, split, sample count, and batch size."""
        super().__init__(**kwargs)
        self._dataset_name = dataset
        self._split = split
        self._num_samples = num_samples
        self._batch_size = batch_size

    def evaluate(  # type: ignore[override]  # pyrefly: ignore[bad-override]
        self,
        model: OliveModelHandler | nn.Module,
        metrics: list[Metric],
        device: Device = Device.CPU,
        execution_providers: str | list[str] | None = None,
    ) -> MetricResult:
        """Evaluate top-k accuracy on the configured HF dataset."""
        del execution_providers  # not used for pure-PyTorch evaluation

        torch_model = _load_model(model)
        torch_device = torch.device("cuda" if device == Device.GPU else "cpu")

        logger.info(
            "ImageNetAccuracyEvaluator: loading %s/%s (up to %d samples)",
            self._dataset_name,
            self._split,
            self._num_samples,
        )
        from datasets import Dataset, load_dataset  # noqa: PLC0415

        ds: Dataset = load_dataset(self._dataset_name, split=self._split)  # type: ignore[assignment]  # pyrefly: ignore[bad-assignment]
        if self._num_samples < len(ds):
            ds = ds.select(range(self._num_samples))

        wnids = ds.features["label"].names
        wnid_to_idx = _wnid_to_imagenet_idx()
        label_map = _build_label_map(wnids, wnid_to_idx)

        top1, top5 = _run_topk(torch_model, ds, label_map, self._batch_size, torch_device)
        logger.info("top-1=%.4f  top-5=%.4f  (n=%d)", top1, top5, len(ds))

        return _build_result(metrics, top1, top5)


_SUB_TYPES = {"top1": True, "top5": True}  # value = higher_is_better


def _build_result(metrics: list[Metric], top1: float, top5: float) -> MetricResult:
    values = {"top1": top1, "top5": top5}
    root: dict[str, SubMetricResult] = {}
    for metric in metrics:
        for sub in metric.sub_types:
            name = sub.name
            if name not in _SUB_TYPES:
                msg = (
                    f"ImageNetAccuracyEvaluator: unknown sub_type {name!r}. "
                    f"Supported: {sorted(_SUB_TYPES)}"
                )
                raise ValueError(msg)
            key = joint_metric_key(metric.name, name)
            root[key] = SubMetricResult(
                value=values[name],
                priority=sub.priority,
                higher_is_better=_SUB_TYPES[name],
            )
    return MetricResult(root=root)


def _load_model(model: OliveModelHandler | nn.Module) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    if isinstance(model, PyTorchModelHandlerBase):
        loaded = model.load_model()
        if not isinstance(loaded, nn.Module):
            msg = f"load_model returned {type(loaded).__name__}, expected nn.Module"
            raise TypeError(msg)
        return loaded
    msg = f"Unsupported model type: {type(model).__name__}"
    raise TypeError(msg)
