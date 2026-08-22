# Default recipe - show available commands
default:
    @just --list

# Install development dependencies
install:
    uv sync --all-groups
    pre-commit install

# Build the Rust extension in development mode
dev:
    uv run maturin develop

# Build the Rust extension in release mode
build:
    uv run maturin develop --release

# Run the fast test suite (skips slow and bench tests)
test: dev
    uv run pytest -m "not slow and not bench"

# Run every test except benchmarks
test-all: dev
    uv run pytest -m "not bench"

# Run benchmark tests
bench: build
    uv run pytest -m bench -s

# Run Rust unit tests
test-rust:
    cargo test --release

# Format code (Rust and Python)
fmt:
    cargo fmt
    uv run ruff format

# Lint code (Rust and Python)
lint:
    cargo clippy -- -D warnings
    uv run ruff check

# Run pre-commit on all files
pre-commit:
    pre-commit run --all-files

# Clean build artifacts
clean:
    cargo clean
    rm -rf target/
    rm -rf dist/
    rm -rf *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Build source and wheel distributions
dist:
    uv build

# Serve documentation locally
docs-serve:
    uv run mkdocs serve

# Build documentation
docs-build:
    uv run mkdocs build

# Deploy documentation to GitHub Pages
docs-deploy:
    uv run mkdocs gh-deploy --force

# Release workflow: format, lint, test, build
release: fmt lint test build dist
    @echo "Release checks passed"

# CI workflow: all checks
ci: pre-commit test-rust test
    @echo "CI checks passed"
