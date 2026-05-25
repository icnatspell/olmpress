# HF path — microsoft/resnet-50 structured pruning + fine-tuning

Structured channel pruning followed by knowledge-distillation fine-tuning of
[microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50), orchestrated
end-to-end by Olive via the `HfModel` handler and
[Torch-Pruning](https://github.com/VainF/Torch-Pruning).

For the torchvision / PyTorchModel equivalent see
[`examples/torch/torchvision-resnet-50/`](../../torch/torchvision-resnet-50/).

## Prerequisites

```bash
uv sync
```

Access to `ILSVRC/imagenet-1k` on HuggingFace Hub is required
(gated dataset — request access at
https://huggingface.co/datasets/ILSVRC/imagenet-1k).

## Run

```bash
uv run chisel run examples/hf/microsoft-resnet-50/workflow.yaml
```

This runs the full pipeline in one shot:

1. **Prune** — LAMP global structured pruning at 10% ratio (`classifier.1` excluded)
2. **Fine-tune** — 5-epoch KD fine-tuning on 5k ImageNet training samples
3. **Evaluate** — acc@1 on 200 ImageNet validation samples

Output model and metrics are written to
`examples/hf/microsoft-resnet-50/outputs/lamp_0.10_kd/`.

## Comparison study

To compare four variants side-by-side (unpruned, pruned-only, pruned + plain CE
fine-tune, pruned + KD fine-tune), run the ablation script:

```bash
uv run python examples/hf/microsoft-resnet-50/compare.py
```

It mutates `workflow.yaml` in memory for each variant and prints a results table.
Fine-tuning is capped to 1 epoch for speed — edit the constant near the top of
`compare.py` for higher-quality numbers.

### Results (1 epoch, 5k train samples, 200 val samples)

| Model                            |   acc@1 |       #params |
|----------------------------------|---------|---------------|
| Unpruned baseline                |  80.50% |    25,557,032 |
| Pruned (no fine-tune)            |  39.50% |    20,437,943 |
| Pruned + plain CE fine-tune      |  39.50% |    20,437,943 |
| Pruned + KD fine-tune            |  42.50% |    20,437,943 |

10% LAMP pruning removes ~20% of parameters. The HF model recovers more slowly
than the torchvision one (~76% baseline vs. ~80%) and needs more than a single
epoch of fine-tuning before plain CE moves the needle. KD provides a small lift
even at 1 epoch because the teacher signal is richer than labels alone.

## File layout

```
examples/hf/microsoft-resnet-50/
├── README.md               # this file
├── workflow.yaml           # Olive workflow: prune → fine-tune → evaluate
├── finetune_script.py      # user script for FineTunePass (KD fine-tuning)
├── eval_data_script.py     # user script for Olive evaluator (ImageNet val)
└── compare.py              # ablation: unpruned vs. pruned vs. fine-tuned
```
