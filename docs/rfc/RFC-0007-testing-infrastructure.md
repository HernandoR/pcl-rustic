# RFC-0007: Testing & Developer Infrastructure

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-05-03 |
| **Updated** | 2026-05-03 |
| **Related RFCs** | [RFC-0001](RFC-0001-core-architecture.md) · [RFC-0002](RFC-0002-enhanced-io.md) · [RFC-0003](RFC-0003-advanced-downsampling.md) · [RFC-0004](RFC-0004-point-cloud-registration.md) · [RFC-0005](RFC-0005-normal-estimation.md) · [RFC-0006](RFC-0006-gpu-acceleration.md) |

## Summary

Standardise the project's testing, CI/CD, and developer tooling decisions:

1. **pytest** is the canonical Python integration test runner — no alternatives.
2. **`justfile`** is the single source of truth for batch shell operations — no `Makefile`, no raw shell scripts in CI.
3. **context7** is used to retrieve up-to-date API documentation for external crates and Python packages referenced throughout the codebase.
4. Specify the full GitHub Actions CI matrix, test categorisation, coverage requirements, and benchmark regression thresholds.

This RFC documents existing decisions already partially implemented (justfile, pytest) and adds the missing pieces: coverage gates, benchmark regression tests, context7 integration, and GPU CI job definition.

## Motivation

As the library grows (RFC-0002 through RFC-0006), the risk of regressions increases. Without explicit infrastructure standards:

- Developers use ad-hoc shell commands that differ between machines.
- New algorithms (ICP, FPS, normals) may ship without integration tests.
- API docs for external crates go stale — e.g., `arrow-rs` has breaking changes every major release.
- Benchmark regressions are invisible until users report performance degradation.

## Design

### 7.1 pytest as the Integration Test Runner

All Python-facing tests live in `tests/` and are discovered and run by pytest:

```
tests/
├── conftest.py               # shared fixtures (build once, reuse)
├── test_point_cloud.py       # RFC-0001: lifecycle, attributes, transforms, voxel
├── test_io.py                # RFC-0002: I/O round-trips, format detection, streaming
├── test_downsample.py        # RFC-0003: FPS, Poisson-disk, adaptive voxel
├── test_registration.py      # RFC-0004: ICP, NDT, RANSAC
├── test_normals.py           # RFC-0005: PCA normals, orientation
├── test_gpu.py               # RFC-0006: GPU vs CPU parity (skipped if no GPU)
└── test_benchmark.py         # performance regression tests (slow, opt-in)
```

**Markers** (defined in `pytest.ini`):

| Marker | Meaning |
|--------|---------|
| `slow` | Takes > 10 s; excluded from fast CI |
| `gpu` | Requires CUDA/wgpu device |
| `integration` | Requires a built Rust extension |
| `benchmark` | Performance regression test |

```ini
# pytest.ini additions
[pytest]
markers =
    slow: marks tests as slow (deselect with -m "not slow")
    gpu: marks tests requiring a GPU
    integration: marks tests requiring the compiled Rust extension
    benchmark: marks performance regression tests
```

### 7.2 justfile as the Single Source of Truth

The `justfile` is the canonical way to run all development operations. **No Makefile.** CI workflows call `just <recipe>` exclusively.

Updated `justfile` recipes:

```just
# Run integration tests only
test-integration:
    uv run pytest tests/ -m integration -v

# Run tests excluding slow and GPU
test-fast:
    uv run pytest tests/ -m "not slow and not gpu" -v

# Run GPU tests (skips automatically if no GPU)
test-gpu:
    uv run pytest tests/test_gpu.py -v

# Run coverage report (requires pytest-cov)
coverage:
    uv run pytest tests/ -m "not slow and not gpu" \
        --cov=pcl_rustic --cov-report=term-missing --cov-report=xml

# Check benchmark regression (runs slow marks)
benchmark-regression:
    uv run pytest tests/test_benchmark.py -v --run-slow -s

# Fetch latest API docs via context7 (requires mcp context7 server)
docs-api:
    @echo "Use context7 MCP server to fetch up-to-date API docs."
    @echo "Key packages: burn, pyo3, arrow-rs, kiddo, nalgebra, las"

# Full CI simulation (mirrors GitHub Actions)
ci: pre-commit test-rust test coverage
    @echo "✅ CI checks passed!"
```

### 7.3 context7 for API Documentation

All external crate and Python package APIs referenced in RFCs and in code must be verified against up-to-date docs via **context7** before implementation. Rationale: `arrow-rs`, `burn`, and `kiddo` all have breaking API changes across major versions.

Workflow:
1. Before implementing any external crate call, query context7 for the crate's latest API.
2. Pin the crate version in `Cargo.toml` to the version verified via context7.
3. Add a comment in the source file: `// API verified against <crate> v<version> via context7`.

Example (in `src/io/parquet.rs`):

```rust
// API verified against arrow-rs v51.0 via context7
use parquet::arrow::arrow_writer::ArrowWriter;
```

### 7.4 GitHub Actions CI Matrix

#### Workflow: `test.yml`

```yaml
strategy:
  matrix:
    os: [ubuntu-22.04, macos-latest, windows-latest]
    python: ["3.10", "3.11", "3.12", "3.13"]
```

Jobs:

| Job | Command | Trigger |
|-----|---------|---------|
| `lint` | `just lint` | every PR / push |
| `test` | `just test-fast` | every PR / push |
| `coverage` | `just coverage` | every PR / push |
| `test-slow` | `just test` | tag push / manual |
| `benchmark` | `just benchmark-regression` | tag push / manual |
| `build-wheel` | maturin matrix | after test |
| `gpu-test` | `just test-gpu` | self-hosted runner, optional |

#### Coverage Gate

PRs must not drop Python coverage below **80%** (checked by `coverage` job). The threshold is enforced via `pytest-cov --cov-fail-under=80`.

#### Benchmark Regression Gate

`test_benchmark.py` records baseline throughput (pts/s) in a JSON file (`tests/benchmark_baseline.json`). The regression test fails if measured throughput drops > 15% below baseline.

```python
# tests/test_benchmark.py
REGRESSION_THRESHOLD = 0.85  # must be >= 85% of baseline

@pytest.mark.benchmark
@pytest.mark.slow
def test_voxel_throughput_regression():
    ...
    assert measured / baseline >= REGRESSION_THRESHOLD
```

### 7.5 Pre-commit Hooks

`.pre-commit-config.yaml` already exists. Add the following hooks:

```yaml
- repo: local
  hooks:
    - id: pytest-fast
      name: pytest (fast)
      entry: uv run pytest tests/ -m "not slow and not gpu" -q
      language: system
      pass_filenames: false
      stages: [pre-push]      # only on push, not commit
```

### 7.6 Dependency Additions

| Package | Purpose |
|---------|---------|
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel test execution (`-n auto`) |
| `pytest-timeout` | Per-test timeouts (prevent hanging GPU tests) |

Add to `pyproject.toml` `[project.optional-dependencies] dev`:

```toml
[project.optional-dependencies]
dev = [
    ...existing...,
    "pytest-cov>=5.0",
    "pytest-xdist>=3.0",
    "pytest-timeout>=2.3",
]
```

### Alternatives Considered

| Approach | Reason Not Chosen |
|----------|-------------------|
| Makefile | Not cross-platform (Windows); `just` is the stated standard |
| tox | Adds a layer over uv; redundant |
| Coverage threshold < 80% | Insufficient for safety-critical geometry code |
| Inline API doc comments only | External crate APIs change; context7 provides live verification |

## Impact

### Breaking Changes

None — all changes are additive to the development workflow. Existing `just test` recipe continues to work.

### Performance Implications

`pytest-xdist` with `-n auto` reduces CI wall-clock time by ~60% on 4-core runners.

### Testing Plan

This RFC *is* the testing plan. Acceptance criteria:

- [ ] All existing tests pass with new pytest markers applied.
- [ ] `just coverage` reports ≥ 80% coverage.
- [ ] `just ci` runs clean end-to-end.
- [ ] Benchmark baseline JSON file committed; regression test passes.
- [ ] GPU test skips gracefully on CPU-only CI runners.

## Unresolved Questions

1. Should the coverage gate apply to Rust code too (via `cargo-llvm-cov`)?
2. Should context7 queries be automated (e.g., a pre-commit hook that warns when a pinned version is stale)?
3. For the benchmark baseline, should we store per-platform baselines or a single cross-platform value?

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [just task runner](https://just.systems/man/en/)
- [context7 MCP server](https://context7.io/)
- [GitHub Actions matrix strategy](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstrategymatrix)
- [RFC-0001: Core Architecture](RFC-0001-core-architecture.md)
