# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.4",
#     "torchvision>=0.19",
#     "transformers>=4.45",
#     "datasets>=3.0",
#     "torchdistill>=0.6",
#     "evaluate>=0.4",
#     "scikit-learn>=1.0",
#     "loguru>=0.7",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = [{index = "pytorch-cu128"}]
# torchvision = [{index = "pytorch-cu128"}]
# ///
"""Fine-tune a pruned HF microsoft/resnet-50 student with optional knowledge distillation.

Preprocessing uses AutoImageProcessor from the HuggingFace Hub model card for
evaluation, matching exactly what the model was trained with.

Two modes share all hyperparameters for a fair comparison:
  kd    (default) — Hinton KD via torchdistill KDLoss; teacher is the unpruned
                    torchvision ResNet50 (IMAGENET1K_V2, frozen).
  plain           — Standard cross-entropy fine-tuning, no teacher.

Usage:
    uv run examples/hf/microsoft-resnet-50/finetune.py --model outputs/lamp_0.10/model/model.pt
    uv run examples/hf/microsoft-resnet-50/finetune.py \
        --model outputs/lamp_0.10/model/model.pt --mode plain
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

import evaluate as _evaluate
import torch
from datasets import load_dataset
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torchdistill.losses.mid_level import KDLoss
from torchdistill.misc.log import MetricLogger, SmoothedValue
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor

MODEL_ID = "microsoft/resnet-50"
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class _StreamingSubset(IterableDataset):
    """Thin wrapper that pulls `num_samples` items from a HuggingFace streaming split."""

    def __init__(self, hf_ds, transform, num_samples: int) -> None:
        self._ds = hf_ds
        self._transform = transform
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Generator[tuple[torch.Tensor, int]]:
        for i, sample in enumerate(self._ds):
            if i >= self._num_samples:
                break
            yield self._transform(sample["image"].convert("RGB")), sample["label"]


def _make_train_loader(num_samples: int, batch_size: int) -> DataLoader:
    logger.info("Streaming ImageNet-1k train split ({} samples, batch {})", num_samples, batch_size)
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )
    # num_workers=0 is required for HuggingFace streaming datasets
    return DataLoader(_StreamingSubset(ds, tf, num_samples), batch_size=batch_size, num_workers=0)


def _eval_accuracy(model_path: str, num_samples: int, batch_size: int) -> float:
    """Evaluate a TorchScript model on ImageNet-1k validation.

    Uses AutoImageProcessor (HF model card preprocessing) and evaluate library,
    mirroring eval.py in this directory.
    """
    logger.info("Loading student from {}", model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    logger.info("Streaming ImageNet-1k validation split ({} samples)", num_samples)
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    all_preds: list[int] = []
    all_refs: list[int] = []
    imgs, labels = [], []
    for i, sample in enumerate(ds):
        if i >= num_samples:
            break
        imgs.append(sample["image"].convert("RGB"))
        labels.append(sample["label"])
        if len(imgs) == batch_size:
            pv = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                all_preds += model(pv).argmax(-1).tolist()
            all_refs += labels
            imgs, labels = [], []
    if imgs:
        pv = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            all_preds += model(pv).argmax(-1).tolist()
        all_refs += labels

    return _evaluate.load("accuracy").compute(predictions=all_preds, references=all_refs)[
        "accuracy"
    ]


def _train(args: argparse.Namespace) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}  mode: {}", device, args.mode)

    # Teacher — KD mode only
    teacher: nn.Module | None = None
    kd_criterion: KDLoss | None = None
    if args.mode == "kd":
        logger.info("Loading teacher (torchvision resnet50, IMAGENET1K_V2 weights)")
        teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
        for p in teacher.parameters():
            p.requires_grad_(requires_grad=False)
        # alpha follows torchdistill's KDLoss convention: alpha weights CE, (1-alpha) weights KL
        kd_criterion = KDLoss(
            student_module_path=".",
            student_module_io="output",
            teacher_module_path=".",
            teacher_module_io="output",
            temperature=args.temperature,
            alpha=args.alpha,
        )
        logger.info(
            "KD config: temperature={}, alpha(CE)={:.2f}, beta(KL)={:.2f}",
            args.temperature,
            args.alpha,
            1.0 - args.alpha,
        )
    else:
        logger.info("Plain fine-tuning — no teacher, cross-entropy loss only")

    ce_criterion = nn.CrossEntropyLoss()

    logger.info("Loading student from {}", args.model)
    student = torch.jit.load(args.model).train().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    logger.info("Optimizer: SGD  lr={}  epochs={}", args.lr, args.epochs)

    loader = _make_train_loader(args.train_samples, args.batch_size)

    for epoch in range(args.epochs):
        student.train()
        logger.info("Epoch {}/{} starting", epoch + 1, args.epochs)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("img/s", SmoothedValue(window_size=10, fmt="{value:.0f}"))
        header = f"Epoch [{epoch + 1}/{args.epochs}]"

        for _images, _targets in metric_logger.log_every(loader, args.log_freq, header):
            images, targets = _images.to(device), _targets.to(device)
            t0 = time.time()

            student_logits = student(images)
            if args.mode == "kd":
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = kd_criterion(
                    {".": {"output": student_logits}},
                    {".": {"output": teacher_logits}},
                    targets,
                )
            else:
                loss = ce_criterion(student_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["img/s"].update(images.shape[0] / (time.time() - t0))

        logger.info(
            "Epoch {}/{} done — avg loss: {:.4f}",
            epoch + 1,
            args.epochs,
            metric_logger.loss.global_avg,
        )
        scheduler.step()

    suffix = "_kd" if args.mode == "kd" else "_ft"
    output_path = args.output or str(Path(args.model).with_stem(Path(args.model).stem + suffix))
    n_params = sum(p.numel() for p in student.parameters())
    torch.jit.save(student, output_path)
    logger.success("Saved fine-tuned student → {}", output_path)
    return output_path, n_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune an olmpress-pruned HF ResNet50 (KD or plain CE)"
    )
    parser.add_argument("--model", required=True, help="Pruned TorchScript .pt file (student)")
    parser.add_argument(
        "--mode",
        choices=["kd", "plain"],
        default="kd",
        help="kd: knowledge distillation from unpruned teacher; plain: cross-entropy only",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for fine-tuned .pt (default: <model>_kd.pt or <model>_ft.pt)",
    )
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--temperature", type=float, default=4.0, help="KD softmax temperature (kd mode only)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="CE loss weight a (torchdistill convention: a*CE + (1-a)*t^2*KL)",
    )
    parser.add_argument("--log-freq", type=int, default=10)
    args = parser.parse_args()

    output_path, n_params = _train(args)

    logger.info("Evaluating on {} validation samples", args.eval_samples)
    acc1 = _eval_accuracy(output_path, args.eval_samples, args.batch_size)
    logger.success("top-1 accuracy: {:.2f}%  (top1={:.4f})", acc1 * 100, acc1)
    print(f"top1={acc1:.4f}", flush=True)
    print(f"params={n_params}", flush=True)

    os._exit(0)  # bypass datasets streaming GC cleanup which hangs on open HTTP connections


if __name__ == "__main__":
    main()
