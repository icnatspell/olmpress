"""Evaluate microsoft/resnet-50 (baseline or pruned) on zh-plus/tiny-imagenet.

Uses ``evaluate.load("accuracy")`` from the HuggingFace evaluate library.
Tiny-imagenet labels are WordNet IDs; the ImageNet-1k class index maps them
to the model's output positions so predictions can be compared to ground truth.

Usage:
    # Evaluate the original microsoft/resnet-50:
    uv run python examples/resnet50_pruning/eval_accuracy.py

    # Evaluate a pruned TorchScript model saved by the Olive workflow:
    uv run python examples/resnet50_pruning/eval_accuracy.py \
        --model workflows/resnet50-pruning/outputs/lamp_0.10/model.pt

    # Quick smoke-test with fewer samples:
    uv run python examples/resnet50_pruning/eval_accuracy.py --num-samples 200
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import evaluate
import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "microsoft/resnet-50"
DATASET = "zh-plus/tiny-imagenet"
SPLIT = "valid"
_IMAGENET_INDEX_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def _build_label_mapping(wnids: list[str]) -> dict[int, int]:
    """Build {imagenet1k_class_idx: tiny_imagenet_int} for the 200 overlapping classes.

    The TF ImageNet class index provides wnid → imagenet1k position, which
    aligns with the model's id2label ordering.
    """
    with urllib.request.urlopen(_IMAGENET_INDEX_URL, timeout=30) as resp:  # noqa: S310
        cls_idx: dict[str, list[str]] = json.loads(resp.read())

    wnid_to_tiny = {w: i for i, w in enumerate(wnids)}
    mapping: dict[int, int] = {}
    for inet_idx_str, (wnid, _) in cls_idx.items():
        if wnid in wnid_to_tiny:
            mapping[int(inet_idx_str)] = wnid_to_tiny[wnid]
    return mapping


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class _HFModel(nn.Module):
    """Wraps AutoModelForImageClassification to return raw logits."""

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._model = AutoModelForImageClassification.from_pretrained(model_id)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._model(pixel_values, return_dict=False)[0]


def _load_model(model_path: str | None) -> nn.Module:
    if model_path:
        return torch.jit.load(model_path)
    return _HFModel(MODEL_ID)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate(  # noqa: PLR0913
    model: nn.Module,
    processor: AutoImageProcessor,
    ds: object,
    label_mapping: dict[int, int],
    batch_size: int,
    device: torch.device,
) -> float:
    """Return top-1 accuracy using evaluate.load('accuracy')."""
    model = model.to(device).eval()
    metric = evaluate.load("accuracy")

    for start in range(0, len(ds), batch_size):  # type: ignore[arg-type]
        batch = ds.select(range(start, min(start + batch_size, len(ds))))  # type: ignore[union-attr]
        images = [item["image"].convert("RGB") for item in batch]
        refs = [item["label"] for item in batch]

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        logits = model(pixel_values)

        # Map imagenet1k class indices to tiny-imagenet indices for comparison.
        top1_inet = logits.argmax(dim=-1).tolist()
        preds = [label_mapping.get(idx, -1) for idx in top1_inet]

        metric.add_batch(predictions=preds, references=refs)

    return metric.compute()["accuracy"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 on tiny-imagenet.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a pruned TorchScript .pt file. Omit to evaluate the baseline.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of validation images to evaluate (default: 2000).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {DATASET}/{SPLIT} …")
    ds = load_dataset(DATASET, split=SPLIT)
    if args.num_samples < len(ds):
        ds = ds.select(range(args.num_samples))
    wnids: list[str] = ds.features["label"].names

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    label_mapping = _build_label_mapping(wnids)
    print(f"  {len(label_mapping)} of {len(wnids)} tiny-imagenet classes mapped")

    model = _load_model(args.model)
    label = args.model or f"baseline ({MODEL_ID})"
    print(f"Evaluating: {label}  [{len(ds)} samples, device={device}]")

    top1 = _evaluate(model, processor, ds, label_mapping, args.batch_size, device)
    print(f"\ntop-1 accuracy: {top1 * 100:.2f}%")
    print(f"top1={top1:.4f}")


if __name__ == "__main__":
    main()
