# HF path — microsoft/resnet-50 structured pruning

One-shot structured channel pruning of
[microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50) loaded via
the Olive `HfModel` handler, backed by
[Torch-Pruning](https://github.com/VainF/Torch-Pruning).

Preprocessing throughout uses `AutoImageProcessor` from the HuggingFace Hub
model card, matching exactly what the model was trained with.

For the pure-PyTorch / torchvision equivalent see
[`examples/torch/torchvision-resnet-50/`](../../torch/torchvision-resnet-50/).

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
uv run olmpress run --config examples/hf/microsoft-resnet-50/workflow.yaml
```

Output: `examples/hf/microsoft-resnet-50/outputs/lamp_0.10/model/model.pt`

Pruning settings:

| Parameter | Value |
|-----------|-------|
| `importance` | `lamp` |
| `pruning_ratio` | `0.10` |
| `global_pruning` | `true` |
| `ignored_layers` | `classifier.1` |

### Step 2 — Evaluate pruned model (before fine-tuning)

```bash
uv run python examples/hf/microsoft-resnet-50/eval.py \
    --model examples/hf/microsoft-resnet-50/outputs/lamp_0.10/model/model.pt \
    --num-samples 200
```

Baseline (unpruned):

```bash
uv run python examples/hf/microsoft-resnet-50/eval.py --num-samples 200
```

### Step 3a — KD fine-tuning (with teacher)

```bash
uv run examples/hf/microsoft-resnet-50/finetune.py \
    --model examples/hf/microsoft-resnet-50/outputs/lamp_0.10/model/model.pt \
    --mode kd --train-samples 5000 --epochs 5
```

Output: `…/model/model_kd.pt`. Evaluation runs automatically at the end.

### Step 3b — Plain fine-tuning (cross-entropy only, for comparison)

```bash
uv run examples/hf/microsoft-resnet-50/finetune.py \
    --model examples/hf/microsoft-resnet-50/outputs/lamp_0.10/model/model.pt \
    --mode plain --train-samples 5000 --epochs 5
```

Output: `…/model/model_ft.pt`.

### Step 4 — Evaluate fine-tuned models

```bash
uv run python examples/hf/microsoft-resnet-50/eval.py \
    --model examples/hf/microsoft-resnet-50/outputs/lamp_0.10/model/model_kd.pt \
    --num-samples 200

uv run python examples/hf/microsoft-resnet-50/eval.py \
    --model examples/hf/microsoft-resnet-50/outputs/lamp_0.10/model/model_ft.pt \
    --num-samples 200
```

## Expected results

Measured on 200 ImageNet-1k validation samples.

| Model | top-1 |
|-------|-------|
| Baseline (microsoft/resnet-50) | ~76% |
| LAMP 10% pruning — one-shot | ~41–42% |
| Plain fine-tuning (5 epochs, 5k samples) | ~55–60% |
| KD fine-tuning (5 epochs, 5k samples) | ~60–65% |

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
examples/hf/microsoft-resnet-50/
├── README.md               # this file
├── workflow.yaml           # HfModel → TorchPruningPass → TorchScript
├── eval.py                 # acc@1 evaluation (baseline or pruned)
└── finetune.py             # KD or plain fine-tuning (standalone, uv run)
```
