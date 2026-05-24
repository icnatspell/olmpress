"""ResNet-50 structured pruning sweep.

Evaluates the baseline model and all method x sparsity combinations defined in
sweep_config.yaml and prints a formatted comparison table.

Usage:
    uv run python examples/resnet50_pruning/run_sweep.py
    uv run python examples/resnet50_pruning/run_sweep.py --config path/to/sweep_config.yaml
    uv run python examples/resnet50_pruning/run_sweep.py --csv results.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.models as tvm
import torchvision.transforms as T  # noqa: N812
import yaml
from torch import nn

import olmpress  # noqa: F401 - registers evaluators/passes

# ---------------------------------------------------------------------------
# Data structures


@dataclass
class Variant:
    """Pruning configuration for one sweep entry."""

    importance: str
    pruning_ratio: float
    calibration_steps: int = 10


@dataclass
class Result:
    """Evaluation result for one model variant."""

    label: str
    importance: str
    pruning_ratio: float
    params: int
    top1: float
    top5: float
    elapsed_s: float


# ---------------------------------------------------------------------------
# Dataset helpers


def _wnid_to_imagenet_idx() -> dict[str, int]:
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read())
    return {v[0]: int(k) for k, v in data.items()}


def preprocess_dataset(
    dataset_name: str,
    split: str,
    num_samples: int,
    transform: T.Compose,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images_tensor [N,3,H,W], labels_tensor [N]) on CPU."""
    from datasets import load_dataset  # noqa: PLC0415

    print(f"Loading {dataset_name}/{split} …")
    ds = load_dataset(dataset_name, split=split)
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    wnids: list[str] = ds.features["label"].names
    wnid_to_idx = _wnid_to_imagenet_idx()
    label_map: list[int | None] = [wnid_to_idx.get(w) for w in wnids]

    imgs: list[torch.Tensor] = []
    labels: list[int] = []
    for item in ds:
        inet_idx = label_map[item["label"]]
        if inet_idx is None:
            continue
        imgs.append(transform(item["image"].convert("RGB")))
        labels.append(inet_idx)

    print(f"  {len(imgs)} images ready ({len(ds) - len(imgs)} skipped — wnid not in index)")
    return torch.stack(imgs), torch.tensor(labels)


# ---------------------------------------------------------------------------
# Evaluation


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (top-1, top-5) accuracy."""
    model.eval()
    model = model.to(device)
    correct1 = correct5 = 0
    n = len(labels)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        imgs_b = images[start:end].to(device)
        gt_b = labels[start:end].to(device)

        logits = model(imgs_b)
        if not isinstance(logits, torch.Tensor):
            logits = logits[0]

        _, top5_preds = logits.topk(5, dim=1)
        correct1 += int((top5_preds[:, 0] == gt_b).sum())
        correct5 += int((top5_preds == gt_b.unsqueeze(1)).any(dim=1).sum())

    return correct1 / n, correct5 / n


# ---------------------------------------------------------------------------
# Pruning


def prune_copy(  # noqa: PLR0913
    base_model: nn.Module,
    variant: Variant,
    example_input: torch.Tensor,
    ignored_layers: list[nn.Module],
    round_to: int | None,
    *,
    global_pruning: bool = False,
) -> nn.Module:
    from olmpress.passes.pytorch.sparsification.structured_pruning import (  # noqa: PLC0415
        prune_model,
    )

    model = copy.deepcopy(base_model).cpu()
    # Re-resolve ignored layers on the copy (same attribute names, new objects).
    ignored_on_copy = [
        getattr(model, name)
        for name in [
            n for n, m in base_model.named_modules() if any(m is ig for ig in ignored_layers)
        ]
    ]
    prune_model(
        model,
        example_inputs=example_input,
        pruning_ratio=variant.pruning_ratio,
        importance=variant.importance,
        ignored_layers=ignored_on_copy,
        round_to=round_to,
        global_pruning=global_pruning,
        calibration_steps=variant.calibration_steps,
    )
    return model


# ---------------------------------------------------------------------------
# Formatting


def _fmt_pct(v: float) -> str:
    return f"{v * 100:5.1f}%"


def _fmt_params(n: int) -> str:
    return f"{n / 1e6:5.1f}M"


def print_table(results: list[Result], baseline_params: int) -> None:
    header = (
        f"{'importance':<16} {'ratio':>6}  "
        f"{'params':>7}  {'retained':>8}  "
        f"{'top-1':>6}  {'top-5':>6}  {'time':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{'':=<{len(header)}}")
    print("  ResNet-50 structured pruning sweep")
    print(f"{'':=<{len(header)}}\n")
    print(header)
    print(sep)
    for r in results:
        retained = r.params / baseline_params
        print(
            f"{r.importance:<16} {_fmt_pct(r.pruning_ratio):>6}  "
            f"{_fmt_params(r.params):>7}  {_fmt_pct(retained):>8}  "
            f"{_fmt_pct(r.top1):>6}  {_fmt_pct(r.top5):>6}  {r.elapsed_s:>5.1f}s"
        )
    print(sep)


def save_csv(results: list[Result], path: str) -> None:
    import csv  # noqa: PLC0415

    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["importance", "pruning_ratio", "params", "top1", "top5", "elapsed_s"],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "importance": r.importance,
                    "pruning_ratio": r.pruning_ratio,
                    "params": r.params,
                    "top1": r.top1,
                    "top5": r.top5,
                    "elapsed_s": r.elapsed_s,
                }
            )
    print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="ResNet-50 pruning sweep.")
    here = Path(__file__).parent
    parser.add_argument(
        "--config",
        default=str(here / "sweep_config.yaml"),
        help="Path to sweep_config.yaml",
    )
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Tiny-ImageNet images are 64x64; direct resize to 224 avoids upsampling artefacts
    # from the standard 256→crop-224 pipeline and gives ~5 pp better accuracy.
    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_cfg = cfg["evaluation"]
    images, labels = preprocess_dataset(
        eval_cfg["dataset"],
        eval_cfg["split"],
        eval_cfg["num_samples"],
        transform,
    )

    # Load base model once; prune copies.
    model_cfg = cfg["model"]
    weights = getattr(tvm.ResNet50_Weights, model_cfg["weights"])
    base_model = tvm.resnet50(weights=weights).eval()
    ignored_layer_names = model_cfg.get("ignored_layers", [])
    ignored_layers = [m for n, m in base_model.named_modules() if n in ignored_layer_names]
    round_to = model_cfg.get("round_to")
    global_pruning = model_cfg.get("global_pruning", False)
    example_input = torch.randn(1, 3, 224, 224)

    baseline_params = sum(p.numel() for p in base_model.parameters())
    results: list[Result] = []

    # --- baseline ---
    print("\nEvaluating baseline …", end=" ", flush=True)
    t0 = time.perf_counter()
    top1, top5 = evaluate(base_model, images, labels, eval_cfg["batch_size"], device)
    elapsed = time.perf_counter() - t0
    results.append(Result("baseline", "baseline", 0.0, baseline_params, top1, top5, elapsed))
    print(f"top-1={top1:.4f}  top-5={top5:.4f}  ({elapsed:.1f}s)")

    # --- pruned variants ---
    variants = [
        Variant(
            importance=imp_cfg["name"],
            pruning_ratio=ratio,
            calibration_steps=imp_cfg.get("calibration_steps", 10),
        )
        for imp_cfg in cfg["importances"]
        for ratio in cfg["pruning_ratios"]
    ]

    for v in variants:
        label = f"{v.importance} {int(v.pruning_ratio * 100)}%"
        print(f"  {label:<28}", end=" ", flush=True)
        t0 = time.perf_counter()
        pruned = prune_copy(
            base_model, v, example_input, ignored_layers, round_to, global_pruning=global_pruning
        )
        params = sum(p.numel() for p in pruned.parameters())
        top1, top5 = evaluate(pruned, images, labels, eval_cfg["batch_size"], device)
        elapsed = time.perf_counter() - t0
        results.append(Result(label, v.importance, v.pruning_ratio, params, top1, top5, elapsed))
        print(f"top-1={top1:.4f}  top-5={top5:.4f}  params={params / 1e6:.1f}M  ({elapsed:.1f}s)")

    print_table(results, baseline_params)

    if args.csv:
        save_csv(results, args.csv)


if __name__ == "__main__":
    main()
