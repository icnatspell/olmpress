# Examples

End-to-end Olive workflows showing how to use chisel's passes and evaluators.
Each example runs in a single command:

```bash
uv run chisel run --config <example>/workflow.yaml
```

## Available examples

| Example | Model | Path |
|---------|-------|------|
| Prune → Fine-tune → Evaluate (HuggingFace) | `microsoft/resnet-50` | [`hf/microsoft-resnet-50/`](hf/microsoft-resnet-50/) |
| Prune → Fine-tune → Evaluate (torchvision) | `torchvision.models.resnet50` | [`torch/torchvision-resnet-50/`](torch/torchvision-resnet-50/) |

Both ResNet50 examples illustrate the same compression flow on the two main
PyTorch entry-points chisel supports (`HfModel` vs `PyTorchModel`).

## Anatomy of an example

Each example directory contains:

- `workflow.yaml` — the Olive workflow chisel runs end-to-end
- `finetune_script.py` — user script for `FineTunePass` (defines `finetune(model, config)`)
- `eval_data_script.py` — user script for Olive's evaluator (data + post-processing)
- `compare.py` — ablation script that runs four variants (unpruned, pruned-only,
  pruned + plain CE, pruned + KD) and prints a single comparison table
- Optional: `load_model.py` — model loader for `PyTorchModel` workflows
- `README.md` — what the example does, prerequisites, expected results
