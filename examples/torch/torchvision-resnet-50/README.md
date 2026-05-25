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
uv run chisel run --config examples/torch/torchvision-resnet-50/workflow.yaml
```

This runs the full pipeline in one shot:

1. **Prune** — LAMP global structured pruning at 10% ratio (`fc` layer excluded)
2. **Fine-tune** — 5-epoch KD fine-tuning on 5k ImageNet training samples
3. **Evaluate** — acc@1 on 200 ImageNet validation samples

Output model and metrics are written to
`examples/torch/torchvision-resnet-50/outputs/lamp_0.10_kd/`.

## Expected results

Measured on 200 ImageNet-1k validation samples.

| Model | top-1 |
|-------|-------|
| Baseline (torchvision ResNet50 IMAGENET1K_V2) | ~80% |
| LAMP 10% pruning — one-shot | ~43–45% |
| KD fine-tuning (5 epochs, 5k samples) | ~62–67% |

## File layout

```
examples/torch/torchvision-resnet-50/
├── README.md               # this file
├── workflow.yaml           # Olive workflow: prune → fine-tune → evaluate
├── load_model.py           # model script (loads torchvision ResNet50)
├── finetune_script.py      # user script for FineTunePass (KD fine-tuning)
└── eval_data_script.py     # user script for Olive evaluator (ImageNet val)
```
