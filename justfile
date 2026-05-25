default:
    @just --list

# Sync dependencies (including dev group)
install:
    uv sync

# Run linting, formatting, and static type checking
check:
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
