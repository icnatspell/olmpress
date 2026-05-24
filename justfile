default:
    @just --list

# Sync dependencies (including dev group)
install:
    uv sync

# Run lint, typecheck, and tests
check: lint typecheck test

# Lint and check formatting
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint and apply formatting
fix:
    uv run ruff check --fix .
    uv run ruff format .

# Type-check with pyrefly
typecheck:
    uv run pyrefly check

# Run tests
test:
    uv run pytest

# Run tests with coverage report
cov:
    uv run pytest --cov --cov-report=term-missing

# Install pre-commit hooks (requires prek installed)
hooks-install:
    prek install

# Run pre-commit hooks on all files
hooks-run:
    prek run --all-files

# Build sdist and wheel
build:
    uv build

# Remove caches and build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
