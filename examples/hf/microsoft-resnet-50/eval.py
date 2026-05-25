"""Evaluate microsoft/resnet-50 (baseline or pruned) on ILSVRC/imagenet-1k.

Preprocessing uses AutoImageProcessor from the HuggingFace Hub model card,
matching exactly what the model was trained with.

Reports acc@1 (via evaluate library).

Usage:
    uv run python examples/hf/microsoft-resnet-50/eval.py
    uv run python examples/hf/microsoft-resnet-50/eval.py --num-samples 200
    uv run python examples/hf/microsoft-resnet-50/eval.py --model outputs/lamp_0.10/model/model.pt
"""

from __future__ import annotations

import argparse
import os

import evaluate as _evaluate
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, pipeline

MODEL_ID = "microsoft/resnet-50"


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
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    all_preds: list[int] = []
    all_refs: list[int] = []

    if args.model:
        model = torch.jit.load(args.model).eval().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        for imgs, labels in _iter_batches(ds, args.batch_size, args.num_samples):
            pv = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                all_preds += model(pv).argmax(-1).tolist()
            all_refs += labels
    else:
        pipe = pipeline("image-classification", model=MODEL_ID, device=device)
        n_params = sum(p.numel() for p in pipe.model.parameters())
        label2id: dict[str, int] = pipe.model.config.label2id
        for imgs, labels in _iter_batches(ds, args.batch_size, args.num_samples):
            results = pipe(imgs)
            all_preds += [label2id[r[0]["label"]] for r in results]
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
