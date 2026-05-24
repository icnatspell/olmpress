# olmpress

[![CI](https://github.com/icnatspell/olmpress/actions/workflows/ci.yml/badge.svg)](https://github.com/icnatspell/olmpress/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20with-uv-261230?logo=uv)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyrefly](https://img.shields.io/badge/checked%20by-pyrefly-1F77B4)](https://github.com/facebook/pyrefly)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Add your description here.

## Installation

```bash
uv add olmpress
```

Or, for local development:

```bash
git clone https://github.com/<you>/olmpress.git
cd olmpress
just install
```

## Usage

```bash
olmpress
```

Or from Python:

```python
from olmpress import main

main()
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) for package management,
[ruff](https://docs.astral.sh/ruff/) for linting and formatting,
[pyrefly](https://github.com/facebook/pyrefly) for type checking,
[pytest](https://docs.pytest.org/) for testing, and
[just](https://github.com/casey/just) as a command runner.

### Common tasks

```bash
just install      # Sync dependencies
just check        # Lint + typecheck + test
just lint         # Ruff lint + format check
just fix          # Auto-fix lint + format
just typecheck    # Run pyrefly
just test         # Run pytest
just cov          # Tests with coverage report
```

Run `just` with no arguments to see the full list.

### Pre-commit hooks

Hooks are configured in `.pre-commit-config.yaml` and run via
[prek](https://github.com/j178/prek) (a faster drop-in for `pre-commit`):

```bash
just hooks-install   # Install git hook
just hooks-run       # Run hooks on all files
```

### Continuous integration

Lint, type check, and tests run on every push and pull request via
GitHub Actions (`.github/workflows/ci.yml`). Coverage must stay at or above
80% (configured in `pyproject.toml`).

## License

[MIT](LICENSE)
