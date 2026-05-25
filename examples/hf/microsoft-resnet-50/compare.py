"""Run prune→fine-tune ablation and print an acc@1 + #params comparison table.

Runs four variants on microsoft/resnet-50:
  1. Unpruned baseline
  2. Pruned only (no fine-tuning)
  3. Pruned + plain cross-entropy fine-tuning
  4. Pruned + knowledge-distillation fine-tuning

Reuses workflow.yaml as the base configuration and mutates it in memory for
each variant. Each variant writes its outputs under outputs/compare_*/.

Run from the project root:
    uv run python examples/hf/microsoft-resnet-50/compare.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml

import chisel  # noqa: F401  (registers chisel passes/evaluators)
from chisel.cli import _collect_pass_package_config

EXAMPLE_DIR = Path(__file__).parent
WORKFLOW_PATH = EXAMPLE_DIR / "workflow.yaml"
OUTPUTS_DIR = EXAMPLE_DIR / "outputs"


def _count_params(model_path: Path) -> int:
    return sum(p.numel() for p in torch.jit.load(str(model_path)).parameters())


def _run_variant(name: str, base_cfg: dict, mutate) -> tuple[float, int]:
    from olive.package_config import OlivePackageConfig
    from olive.workflows import run as olive_run

    cfg = copy.deepcopy(base_cfg)
    mutate(cfg)
    output_dir = OUTPUTS_DIR / name
    cfg["engine"]["output_dir"] = str(output_dir)

    pkg = OlivePackageConfig.load_default_config().model_dump()
    pkg["passes"].update(_collect_pass_package_config())
    olive_run(cfg, package_config=pkg)

    metrics = json.loads((output_dir / "metrics.json").read_text())
    acc = metrics["accuracy-accuracy_score"]["value"]
    params = _count_params(output_dir / "model.pt")
    return acc, params


def _evaluate_baseline() -> tuple[float, int]:
    sys.path.insert(0, str(EXAMPLE_DIR))
    from eval_data_script import load_imagenet_val  # type: ignore[import-not-found]
    from transformers import AutoModelForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    model = model.eval().to(device)

    data = load_imagenet_val(num_samples=200)
    correct = 0
    with torch.no_grad():
        for tensor, label in data:
            logits = model(pixel_values=tensor.unsqueeze(0).to(device)).logits
            pred = logits.argmax(-1).item()
            correct += int(pred == label)

    return correct / len(data), sum(p.numel() for p in model.parameters())


def _strip_finetune(cfg: dict) -> None:
    del cfg["passes"]["finetuning"]


def _plain_finetune(cfg: dict) -> None:
    cfg["passes"]["finetuning"]["config"]["finetune_config"]["teacher_model"] = None


def _kd_finetune(cfg: dict) -> None:
    """Keep KD defaults (teacher_model=IMAGENET1K_V2)."""


def main() -> None:
    base = yaml.safe_load(WORKFLOW_PATH.read_text())
    results: list[tuple[str, float, int]] = []

    print("=" * 60, flush=True)
    print("Variant 1/4: Unpruned baseline", flush=True)
    print("=" * 60, flush=True)
    acc, params = _evaluate_baseline()
    results.append(("Unpruned baseline", acc, params))

    print("=" * 60, flush=True)
    print("Variant 2/4: Pruned (no fine-tune)", flush=True)
    print("=" * 60, flush=True)
    acc, params = _run_variant("compare_pruned", base, _strip_finetune)
    results.append(("Pruned (no fine-tune)", acc, params))

    print("=" * 60, flush=True)
    print("Variant 3/4: Pruned + plain CE fine-tune", flush=True)
    print("=" * 60, flush=True)
    acc, params = _run_variant("compare_plain", base, _plain_finetune)
    results.append(("Pruned + plain CE fine-tune", acc, params))

    print("=" * 60, flush=True)
    print("Variant 4/4: Pruned + KD fine-tune", flush=True)
    print("=" * 60, flush=True)
    acc, params = _run_variant("compare_kd", base, _kd_finetune)
    results.append(("Pruned + KD fine-tune", acc, params))

    print()
    print("## Results")
    print()
    print(f"| {'Model':<32} | {'acc@1':>7} | {'#params':>13} |")
    print(f"|{'-' * 34}|{'-' * 9}|{'-' * 15}|")
    for name, acc, params in results:
        print(f"| {name:<32} | {acc * 100:>6.2f}% | {params:>13,} |")


if __name__ == "__main__":
    main()
