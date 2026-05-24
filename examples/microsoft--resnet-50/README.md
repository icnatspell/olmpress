# microsoft/resnet-50 structured pruning

One-shot structured channel pruning of [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)
via the olmpress `TorchPruningPass` backed by [Torch-Pruning](https://github.com/VainF/Torch-Pruning).

Two workflow variants are provided:

| Variant | Input model type | Output |
|---------|-----------------|--------|
| `hf/` | `HfModel` (loaded directly from HuggingFace Hub) | TorchScript `.pt` |
| `pytorch/` | `PyTorchModel` (custom loader in `resnet50_model.py`) | TorchScript `.pt` |

Both produce a `PyTorchModelHandler` wrapping a TorchScript file that can be
loaded with `torch.jit.load` and evaluated with `eval_accuracy.py`.

## Prerequisites

```
uv sync
```

Access to `ILSVRC/imagenet-1k` on HuggingFace Hub is required for evaluation
(gated dataset — request access at https://huggingface.co/datasets/ILSVRC/imagenet-1k).

## Running the pruning workflow

All commands run from the **project root**.

### HfModel path

```bash
uv run olmpress run --config examples/microsoft--resnet-50/hf/workflow.yaml
```

Output: `examples/microsoft--resnet-50/hf/outputs/lamp_0.10/`

### PyTorchModel path

```bash
uv run olmpress run --config examples/microsoft--resnet-50/pytorch/workflow.yaml
```

Output: `examples/microsoft--resnet-50/pytorch/outputs/lamp_0.10/`

Both workflows use these pruning settings by default:

| Parameter | Value |
|-----------|-------|
| `importance` | `lamp` |
| `pruning_ratio` | `0.10` (10% channels removed per layer) |
| `global_pruning` | `true` |
| `ignored_layers` | `classifier.1` (HF path only) |

## Evaluating accuracy

The `eval_accuracy.py` script evaluates either the baseline HF model or a
pruned TorchScript `.pt` file on the ImageNet-1k validation set.

### Baseline (no pruning)

```bash
uv run python examples/microsoft--resnet-50/eval_accuracy.py --num-samples 200
```

### Pruned model (HF path)

```bash
uv run python examples/microsoft--resnet-50/eval_accuracy.py \
    --model examples/microsoft--resnet-50/hf/outputs/lamp_0.10/model/model.pt \
    --num-samples 200
```

### Pruned model (PyTorchModel path)

```bash
uv run python examples/microsoft--resnet-50/eval_accuracy.py \
    --model examples/microsoft--resnet-50/pytorch/outputs/lamp_0.10/model/model.pt \
    --num-samples 200
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(none)* | Path to pruned TorchScript `.pt`. Omit for baseline. |
| `--num-samples` | `200` | Number of validation images to evaluate. |
| `--batch-size` | `64` | Batch size for inference. |

## Expected results

Measured on 200 samples from the ImageNet-1k validation set (no fine-tuning).

| Model | Top-1 accuracy |
|-------|---------------|
| Baseline (microsoft/resnet-50) | ~76% |
| LAMP 10% global pruning (one-shot) | ~41–42% |

The large accuracy drop is expected for one-shot structured pruning without
fine-tuning. Fine-tuning after pruning typically recovers most of the accuracy.
A 10% pruning ratio is aggressive for LAMP in global mode — try 5% for a
smaller drop, or fine-tune the pruned model to recover accuracy.

## File layout

```
examples/microsoft--resnet-50/
├── README.md               # this file
├── eval_accuracy.py        # evaluation script (baseline and pruned)
├── resnet50_model.py       # Olive model script for PyTorchModel path
├── hf/
│   └── workflow.yaml       # HfModel → TorchPruningPass → TorchScript
└── pytorch/
    └── workflow.yaml       # PyTorchModel → TorchPruningPass → TorchScript
```
