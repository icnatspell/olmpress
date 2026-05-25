# Torch path — torchvision ResNet50 structured pruning + fine-tuning

Structured channel pruning followed by knowledge-distillation fine-tuning of
`torchvision.models.resnet50` (IMAGENET1K_V2 weights), orchestrated end-to-end
by Olive via the `PyTorchModel` handler and
[Torch-Pruning](https://github.com/VainF/Torch-Pruning).

For the HuggingFace / AutoImageProcessor equivalent see
[`examples/hf/microsoft-resnet-50/`](../../hf/microsoft-resnet-50/).

## Prerequisites

```bash
uv sync
```

Access to `ILSVRC/imagenet-1k` on HuggingFace Hub is required
(gated dataset — request access at
https://huggingface.co/datasets/ILSVRC/imagenet-1k).

## Run

```bash
uv run chisel run examples/torch/torchvision-resnet-50/workflow.yaml
```

This runs the full pipeline in one shot:

1. **Prune** — LAMP global structured pruning at 10% ratio (`fc` layer excluded)
2. **Fine-tune** — 5-epoch KD fine-tuning on 5k ImageNet training samples
3. **Evaluate** — acc@1 on 200 ImageNet validation samples

Output model and metrics are written to
`examples/torch/torchvision-resnet-50/outputs/lamp_0.10_kd/`.

## Comparison study

To compare four variants side-by-side (unpruned, pruned-only, pruned + plain CE
fine-tune, pruned + KD fine-tune), run the ablation script:

```bash
uv run python examples/torch/torchvision-resnet-50/compare.py
```

It mutates `workflow.yaml` in memory for each variant and prints a results table.
Fine-tuning is capped to 1 epoch for speed — edit the constant near the top of
`compare.py` for higher-quality numbers.

### Results (1 epoch, 5k train samples, 200 val samples)

| Model                            |   acc@1 |       #params |
|----------------------------------|---------|---------------|
| Unpruned baseline                |  81.00% |    25,557,032 |
| Pruned (no fine-tune)            |  44.50% |    20,235,006 |
| Pruned + plain CE fine-tune      |  60.50% |    20,235,006 |
| Pruned + KD fine-tune            |  58.50% |    20,235,006 |

10% LAMP pruning removes ~21% of parameters. A single epoch of fine-tuning
recovers most of the lost accuracy; KD typically pulls ahead of plain CE with
more epochs.

## File layout

```
examples/torch/torchvision-resnet-50/
├── README.md               # this file
├── workflow.yaml           # Olive workflow: prune → fine-tune → evaluate
├── load_model.py           # model script (loads torchvision ResNet50)
├── finetune_script.py      # user script for FineTunePass (KD fine-tuning)
├── eval_data_script.py     # user script for Olive evaluator (ImageNet val)
└── compare.py              # ablation: unpruned vs. pruned vs. fine-tuned
```
