# Torch path — torchvision ResNet50 structured pruning

One-shot structured channel pruning of
`torchvision.models.resnet50` (IMAGENET1K_V2 weights) loaded via the Olive
`PyTorchModel` handler, backed by
[Torch-Pruning](https://github.com/VainF/Torch-Pruning).

Preprocessing throughout uses standard torchvision validation transforms
(Resize 256 → CenterCrop 224 → Normalize), matching torchvision's training
recipe — no HuggingFace transformers dependency required in this path.

For the HuggingFace / AutoImageProcessor equivalent see
[`examples/hf/microsoft-resnet-50/`](../../hf/microsoft-resnet-50/).

## Prerequisites

```bash
uv sync
```

Access to `ILSVRC/imagenet-1k` on HuggingFace Hub is required
(gated dataset — request access at
https://huggingface.co/datasets/ILSVRC/imagenet-1k).

## Full workflow

All commands run from the **project root**.

### Step 1 — Prune

```bash
uv run olmpress run --config examples/torch/torchvision-resnet-50/workflow.yaml
```

Output: `examples/torch/torchvision-resnet-50/outputs/lamp_0.10/model/model.pt`

Pruning settings:

| Parameter | Value |
|-----------|-------|
| `importance` | `lamp` |
| `pruning_ratio` | `0.10` |
| `global_pruning` | `true` |
| `ignored_layers` | `[]` (torchvision fc layer is prunable) |

### Step 2 — Evaluate pruned model (before fine-tuning)

```bash
uv run python examples/torch/torchvision-resnet-50/eval.py \
    --model examples/torch/torchvision-resnet-50/outputs/lamp_0.10/model/model.pt \
    --num-samples 200
```

Baseline (unpruned torchvision ResNet50):

```bash
uv run python examples/torch/torchvision-resnet-50/eval.py --num-samples 200
```

### Step 3a — KD fine-tuning (with teacher)

```bash
uv run examples/torch/torchvision-resnet-50/finetune.py \
    --model examples/torch/torchvision-resnet-50/outputs/lamp_0.10/model/model.pt \
    --mode kd --train-samples 5000 --epochs 5
```

Output: `…/model/model_kd.pt`. Evaluation runs automatically at the end.

### Step 3b — Plain fine-tuning (cross-entropy only, for comparison)

```bash
uv run examples/torch/torchvision-resnet-50/finetune.py \
    --model examples/torch/torchvision-resnet-50/outputs/lamp_0.10/model/model.pt \
    --mode plain --train-samples 5000 --epochs 5
```

Output: `…/model/model_ft.pt`.

### Step 4 — Evaluate fine-tuned models

```bash
uv run python examples/torch/torchvision-resnet-50/eval.py \
    --model examples/torch/torchvision-resnet-50/outputs/lamp_0.10/model/model_kd.pt \
    --num-samples 200

uv run python examples/torch/torchvision-resnet-50/eval.py \
    --model examples/torch/torchvision-resnet-50/outputs/lamp_0.10/model/model_ft.pt \
    --num-samples 200
```

## Expected results

Measured on 200 ImageNet-1k validation samples.

| Model | top-1 |
|-------|-------|
| Baseline (torchvision ResNet50 IMAGENET1K_V2) | ~80% |
| LAMP 10% pruning — one-shot | ~43–45% |
| Plain fine-tuning (5 epochs, 5k samples) | ~57–62% |
| KD fine-tuning (5 epochs, 5k samples) | ~62–67% |

Note: torchvision IMAGENET1K_V2 weights are slightly stronger than the HF
microsoft/resnet-50 weights, so baseline accuracy is a few points higher.

## `finetune.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Pruned `.pt` from the Olive workflow |
| `--mode` | `kd` | `kd`: KD from unpruned teacher; `plain`: CE only |
| `--output` | `<model>_kd.pt` / `<model>_ft.pt` | Output path |
| `--train-samples` | `1000` | ImageNet training images to stream |
| `--eval-samples` | `200` | ImageNet validation images to evaluate on |
| `--batch-size` | `32` | Batch size |
| `--epochs` | `3` | Training epochs |
| `--lr` | `0.001` | SGD learning rate (cosine-annealed) |
| `--temperature` | `4.0` | KD softmax temperature τ (kd only) |
| `--alpha` | `0.5` | CE weight α in `α·CE + (1-α)·τ²·KL` (kd only) |
| `--log-freq` | `10` | Batch logging frequency |

## File layout

```
examples/torch/torchvision-resnet-50/
├── README.md               # this file
├── workflow.yaml           # PyTorchModel → TorchPruningPass → TorchScript
├── load_model.py           # Olive model script (loads torchvision ResNet50)
├── eval.py                 # acc@1 evaluation (baseline or pruned)
└── finetune.py             # KD or plain fine-tuning (standalone, uv run)
```
