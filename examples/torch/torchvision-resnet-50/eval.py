"""Evaluate torchvision ResNet50 (baseline or pruned) on ILSVRC/imagenet-1k.

Preprocessing uses standard torchvision validation transforms (Resize 256 →
CenterCrop 224 → Normalize), matching torchvision's training recipe.

Reports acc@1 (via evaluate library).

Usage:
    uv run python examples/torch/torchvision-resnet-50/eval.py
    uv run python examples/torch/torchvision-resnet-50/eval.py --num-samples 200
    uv run python examples/torch/torchvision-resnet-50/eval.py \
        --model outputs/lamp_0.10/model/model.pt
"""

from __future__ import annotations

import argparse
import os

import evaluate as _evaluate
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


def _iter_batches(ds, batch_size: int, num_samples: int):
    imgs, labels = [], []
    for sample in ds.take(num_samples):
        imgs.append(sample["image"].convert("RGB"))
        labels.append(sample["label"])
        if len(imgs) == batch_size:
            yield imgs, labels
            imgs, labels = [], []
    if imgs:
        yield imgs, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Pruned TorchScript .pt file from Olive.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    all_preds: list[int] = []
    all_refs: list[int] = []

    if args.model:
        model = torch.jit.load(args.model).eval().to(device)
    else:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device)

    n_params = sum(p.numel() for p in model.parameters())

    for imgs, labels in _iter_batches(ds, args.batch_size, args.num_samples):
        batch = torch.stack([_VAL_TRANSFORM(img) for img in imgs]).to(device)
        with torch.no_grad():
            all_preds += model(batch).argmax(-1).tolist()
        all_refs += labels

    acc1 = _evaluate.load("accuracy").compute(predictions=all_preds, references=all_refs)[
        "accuracy"
    ]
    print(f"top-1 accuracy: {acc1 * 100:.2f}%", flush=True)
    print(f"top1={acc1:.4f}", flush=True)
    print(f"params={n_params}", flush=True)

    os._exit(0)  # bypass datasets streaming GC cleanup which hangs on open HTTP connections


if __name__ == "__main__":
    main()
