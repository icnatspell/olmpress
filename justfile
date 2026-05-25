default:
    @just --list

# Sync dependencies (including dev group)
install:
    uv sync

# Run lint and tests
check: lint test

# Lint, check formatting, and type-check
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyrefly check

# Auto-fix lint and apply formatting, then type-check
fix:
    uv run ruff check --fix .
    uv run ruff format .
    uv run pyrefly check

# Run tests with coverage report
test:
    uv run pytest --cov --cov-report=term-missing

# Install pre-commit hooks and run them on all files
hooks:
    prek install
    prek run --all-files

# Build sdist and wheel
build:
    uv build

# Remove caches and build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
