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
uv run chisel run --config examples/hf/microsoft-resnet-50/workflow.yaml
```

This runs the full pipeline in one shot:

1. **Prune** — LAMP global structured pruning at 10% ratio (`classifier.1` excluded)
2. **Fine-tune** — 5-epoch KD fine-tuning on 5k ImageNet training samples
3. **Evaluate** — acc@1 on 200 ImageNet validation samples

Output model and metrics are written to
`examples/hf/microsoft-resnet-50/outputs/lamp_0.10_kd/`.

## Expected results

Measured on 200 ImageNet-1k validation samples.

| Model | top-1 |
|-------|-------|
| Baseline (microsoft/resnet-50) | ~76% |
| LAMP 10% pruning — one-shot | ~41–42% |
| KD fine-tuning (5 epochs, 5k samples) | ~40–50% |

## File layout

```
examples/hf/microsoft-resnet-50/
├── README.md               # this file
├── workflow.yaml           # Olive workflow: prune → fine-tune → evaluate
├── finetune_script.py      # user script for FineTunePass (KD fine-tuning)
└── eval_data_script.py     # user script for Olive evaluator (ImageNet val)
```
