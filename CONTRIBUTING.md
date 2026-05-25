# Contributing to chisel

Thanks for your interest! This guide covers the basics for getting set up and
sending a change.

## Development setup

```bash
git clone https://github.com/icnatspell/chisel.git
cd chisel
uv sync
```

Run `just` (no arguments) to see all available recipes:

- `just check` — lint, format, and type-check
- `just test`  — pytest with coverage
- `just hooks` — install and run pre-commit hooks on all files
- `just build` — build sdist and wheel

## Workflow

1. Fork the repo and create a topic branch off `main`.
2. Make focused changes — see [`CLAUDE.md`](.claude/CLAUDE.md) for the surgical
   editing guidelines this project follows.
3. Add or update tests in `tests/`. Coverage must stay at or above 80%.
4. Run `just check` and `just test` locally.
5. Open a PR. CI must be green before review.

## Conventions

- **Commit messages**: conventional-commits flavored, lower-case subject. See
  `git log` for examples. Multi-line bodies are encouraged when the *why* needs
  explaining.
- **Code style**: enforced by ruff (see `pyproject.toml`). No manual formatting.
- **Types**: enforced by pyrefly. Avoid `# type: ignore` where a real fix is
  possible.
- **Docstrings**: required on public modules, classes, and functions (ruff `D`
  rules). Lead with a one-line summary.
- **No comments** that restate what the code does — only WHY when the reason is
  non-obvious (a constraint, an invariant, a workaround for a bug).

## Adding a new Olive pass or evaluator

1. Implement it under `src/chisel/passes/...` or `src/chisel/evaluators/...`.
2. Export it from the relevant `__init__.py` and add the name to `__all__`.
3. The CLI auto-registers everything in `chisel.passes.__all__` — no manual
   wiring needed.
4. Add tests under `tests/` (mirror the source layout).
5. Add an example under `examples/` if it changes the user-facing flow.
6. Update `CHANGELOG.md` under `[Unreleased]`.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`.
