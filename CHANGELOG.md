# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `FineTunePass`: standalone Olive pass that delegates fine-tuning to a
  user-supplied `finetune(model, config) -> model` function via the user-script
  pattern. Wraps the call with an `atexit` workaround so user scripts don't have
  to handle HuggingFace streaming-dataset shutdown hangs.
- Olive-native evaluation in both ResNet50 examples (acc@1, multiclass).
- `chisel list` command — prints chisel-registered passes and evaluators.
- Positional `chisel run path/to/config.yaml` alongside the existing `--config`.
- `--version` flag on the CLI.
- `chisel.__version__` attribute (read from package metadata).
- PyPI metadata: keywords, classifiers, `[project.urls]`.
- `examples/README.md` as an index for the examples directory.
- `compare.py` ablation script in both ResNet50 examples — runs four variants
  (unpruned, pruned, pruned + plain CE, pruned + KD) and prints an
  acc@1 + #params comparison table.
- Pre-commit hooks for `codespell` and `pyrefly`.
- Reusable composite GitHub Action for CI setup.
- Issue and pull-request templates.

### Changed

- Restructured `chisel/__init__.py` so importing the package no longer pulls in
  the CLI. The CLI lives in `chisel.cli`. `chisel.main` is preserved for
  backwards compatibility.
- Lowered the Python floor from 3.13 to 3.11.
- Bumped the `olive-ai` floor from `>=0.9` to `>=0.12` to reflect the features
  actually used.
- Examples no longer need to register the `atexit(os._exit(0))` workaround —
  `FineTunePass` handles it.
- Both ResNet50 examples were collapsed onto a single end-to-end Olive workflow
  (prune → fine-tune → evaluate) with one command. The previous standalone
  `eval.py`, `finetune.py`, and example `justfile` were removed.

### Fixed

- Cleaned up the `config` annotation on `TorchPruningPass._run_for_config`
  (`type[BasePassConfig]` → `Any`). The previous annotation was misleading
  because the runtime value is an instance, not a type. Olive's parent class has
  a typing bug in this signature, so `Any` is the cleanest workaround until it
  is fixed upstream.
