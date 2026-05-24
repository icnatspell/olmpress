"""Structured pruning of a pretrained ResNet-50.

Usage:
    uv run python examples/resnet50_pruning/prune_resnet50.py
    uv run python examples/resnet50_pruning/prune_resnet50.py --importance taylor --ratio 0.4
    uv run python examples/resnet50_pruning/prune_resnet50.py --importance obdc --ratio 0.3

The script prunes a torchvision ResNet-50 (pretrained on ImageNet), reports the
parameter and FLOPs reduction, and saves the pruned model to disk.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torchvision.models as tvm
from torch import nn

import olmpress  # noqa: F401 - registers passes/evaluators
from olmpress.passes.pytorch.sparsification.structured_pruning import prune_model

# ---------------------------------------------------------------------------
# Helpers


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _infer_latency_ms(model: nn.Module, x: torch.Tensor, runs: int = 20) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warm-up
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        return (time.perf_counter() - t0) / runs * 1000


def _top1(model: nn.Module, x: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune a pretrained ResNet-50.")
    parser.add_argument("--ratio", type=float, default=0.5, help="Pruning ratio (0.0-1.0).")
    parser.add_argument(
        "--importance",
        default="magnitude",
        choices=["magnitude", "group_magnitude", "lamp", "fpgm", "taylor", "hessian", "obdc"],
        help="Importance criterion.",
    )
    parser.add_argument("--iterative-steps", type=int, default=1)
    parser.add_argument("--round-to", type=int, default=None)
    parser.add_argument("--global-pruning", action="store_true")
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=10,
        help="Backward passes for gradient-based importances.",
    )
    parser.add_argument("--output", default="pruned_resnet50.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pretrained model and a representative input.
    print("Loading pretrained ResNet-50 ...")
    model: nn.Module = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    example_input = torch.randn(1, 3, 224, 224, device=device)

    # Ignore the classification head so output dim is preserved.
    ignored = [model.fc]

    params_before = _param_count(model)
    latency_before = _infer_latency_ms(model, example_input)

    # Quick accuracy proxy: random batch (just to show the pipeline works).
    fake_images = torch.randn(8, 3, 224, 224, device=device)
    fake_labels = torch.randint(0, 1000, (8,), device=device)
    acc_before = _top1(model, fake_images, fake_labels)

    print(f"Before pruning: {params_before:,} params | {latency_before:.1f} ms/forward")

    print(f"Pruning with importance={args.importance!r}, ratio={args.ratio} ...")
    prune_model(
        model,
        example_inputs=example_input,
        pruning_ratio=args.ratio,
        importance=args.importance,
        iterative_steps=args.iterative_steps,
        ignored_layers=ignored,
        round_to=args.round_to,
        global_pruning=args.global_pruning,
        calibration_steps=args.calibration_steps,
    )

    params_after = _param_count(model)
    latency_after = _infer_latency_ms(model, example_input)
    acc_after = _top1(model, fake_images, fake_labels)

    reduction = 1.0 - params_after / params_before
    speedup = latency_before / latency_after if latency_after > 0 else float("nan")

    print(f"After pruning:  {params_after:,} params | {latency_after:.1f} ms/forward")
    print(f"Param reduction: {reduction:.1%} | Latency speedup: {speedup:.2f}x")
    print(f"Accuracy proxy (random batch) before={acc_before:.2f} after={acc_after:.2f}")

    out = Path(args.output)
    scripted = torch.jit.trace(model.cpu(), example_input.cpu())
    torch.jit.save(scripted, str(out))
    print(f"Saved pruned model to {out}")


if __name__ == "__main__":
    main()
