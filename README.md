<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/logo-light-text.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/source/images/logo-dark-text.png">
    <img alt="chisel" src="docs/source/images/logo-dark-text.png" height="300" style="max-width: 100%;">
  </picture>

[![CI](https://github.com/icnatspell/chisel/actions/workflows/ci.yml/badge.svg)](https://github.com/icnatspell/chisel/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-261230?logo=uv)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyrefly](https://img.shields.io/badge/checked%20by-pyrefly-1F77B4)](https://github.com/facebook/pyrefly)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
</div>

`chisel` is a model compression toolkit for PyTorch and Hugging Face Transformers that helps you sculpt leaner, faster models for efficient inference at the edge.

It exposes a set of composable compression passes, including quantization, pruning, and low-rank decomposition, that can be applied individually or orchestrated together into a compression pipeline via a simple YAML workflow. Under the hood, `chisel` builds on Microsoft's Olive for workflow orchestration. Each pass is designed to be modular, with first-class support for PyTorch and Transformers models.

`chisel` also provides built-in evaluators to measure the impact of each compression step on model size and accuracy, so you always know the cost of every cut.
Longer term, `chisel` aims to support export to edge runtimes such as ONNX and ExecuTorch, bringing the full compression-to-deployment pipeline under one roof.

## Installation

`chisel` is not yet published on PyPI. Install the latest from source:

```bash
uv add git+https://github.com/icnatspell/chisel
```

Or clone and `uv sync` (see [Development](#development) below).

## Usage

`chisel` is driven by Olive workflow configs. The `chisel run` command is a thin wrapper around `olive run` that ensures chisel's passes and evaluators are registered before Olive starts:

```bash
chisel run --config path/to/workflow.yaml
```

A minimal structured-pruning config looks like:

```yaml
input_model:
  type: HfModel
  model_path: microsoft/resnet-50
  task: image-classification

passes:
  pruning:
    type: TorchPruningPass
    config:
      pruning_ratio: 0.10
      importance: lamp
      global_pruning: true

engine:
  output_dir: outputs/pruned
```

See [`examples/`](examples/) for complete end-to-end workflows including evaluation and knowledge-distillation fine-tuning.

## Examples

| Example | Model | What it shows |
|---------|-------|---------------|
| [`examples/hf/microsoft-resnet-50/`](examples/hf/microsoft-resnet-50/) | `microsoft/resnet-50` | Prune → Eval → Fine-tune (Plain, KD) |
| [`examples/torch/torchvision-resnet-50/`](examples/torch/torchvision-resnet-50/) | `torchvision.models.resnet50` | Prune → Eval → Fine-tune (Plain, KD) |

## Development

```bash
git clone https://github.com/icnatspell/chisel.git && cd chisel && uv sync
```

```bash
just check   # lint, format, type-check
just test    # pytest with coverage
just build   # build sdist + wheel
```

Run `just` with no arguments to see all tasks.

### Pre-commit hooks

```bash
just hooks   # install hooks and run on all files
```

### Continuous integration

Lint, type-check, and tests run on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`). Coverage must stay at or above 80%.

## License

[MIT](LICENSE)
