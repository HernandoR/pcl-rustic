# RFC-0007 — Testing & CI Infrastructure Enhancement

| Field | Value |
|-------|-------|
| RFC | 0007 |
| Status | Open for Review |
| Author | liuzhen19 |
| Created | 2026-05-03 |
| Updated | 2026-05-03 |

## Summary

Formalise the testing conventions, CI pipeline, and task-runner structure for PCL Rustic. This RFC codifies the use of **pytest** for all integration tests (including CI), **justfile** for all batch shell operations, and introduces a layered test architecture with markers for fast/slow/GPU tests.

## Motivation

As the library grows with RFC-0002 through RFC-0006, the testing infrastructure must scale to:

- Keep CI fast (< 5 min for the standard matrix) while allowing expensive tests to run on demand.
- Provide a consistent developer experience across platforms (Linux, macOS, Windows).
- Ensure GPU-specific tests do not block standard CI.
- Give new contributors a clear mental model of "how to add a test."

Currently:
- `pytest.ini` exists but markers are not fully documented.
- The justfile has basic recipes but lacks recipes for new feature areas (registration, normals, segmentation).
- There is no formal policy on test file naming or fixture placement.

## Design

### Test architecture

```
tests/
├── conftest.py              # shared fixtures (session/module/function scope)
├── fixtures/                # large fixture files (synthetic point clouds, LAS samples)
│   ├── small_cloud.npy      # 1 K points, f32
│   ├── medium_cloud.npy     # 100 K points, f32
│   └── plane_cloud.npy      # flat plane for normal/segmentation tests
├── test_point_cloud.py      # core API tests (lifecycle, attributes, transforms)
├── test_downsampling.py     # voxel, FPS, normal-based (RFC-0003)
├── test_registration.py     # ICP, NDT (RFC-0004)
├── test_normals.py          # normal estimation (RFC-0005)
├── test_segmentation.py     # RANSAC, clustering (RFC-0006)
├── test_backend.py          # backend selection (RFC-0002)
├── test_io.py               # LAS, CSV, Parquet read/write
├── test_interop.py          # NumPy round-trip, dtype validation
└── test_benchmark.py        # performance benchmarks (marked @pytest.mark.slow)
```

### Pytest markers

Define all markers in `pytest.ini`:

```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with -m "not slow")
    gpu: marks tests requiring a GPU (skip if GPU unavailable)
    benchmark: performance benchmarks (use with -s for output)
    integration: end-to-end tests that build the Rust extension
    smoke: minimal subset for pre-commit / fast feedback
```

### conftest.py — shared fixtures

```python
# tests/conftest.py
import numpy as np
import pytest
from pcl_rustic import PointCloud

def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run slow tests")
    parser.addoption("--run-gpu", action="store_true", default=False,
                     help="Run GPU tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="use --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-gpu"):
        skip_gpu = pytest.mark.skip(reason="use --run-gpu to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

@pytest.fixture(scope="session")
def small_cloud() -> PointCloud:
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((1_000, 3)).astype(np.float32)
    return PointCloud.from_xyz(xyz)

@pytest.fixture(scope="session")
def medium_cloud() -> PointCloud:
    rng = np.random.default_rng(1)
    xyz = rng.standard_normal((100_000, 3)).astype(np.float32)
    return PointCloud.from_xyz(xyz)

@pytest.fixture(scope="session")
def flat_plane() -> PointCloud:
    rng = np.random.default_rng(2)
    xy = rng.uniform(-10, 10, (5_000, 2)).astype(np.float32)
    z = np.zeros((5_000, 1), dtype=np.float32)
    xyz = np.concatenate([xy, z], axis=1)
    return PointCloud.from_xyz(xyz)
```

### Justfile — complete recipe set

```just
# Run all tests (skipping slow and GPU by default)
test:
    uv run pytest tests/ -v

# Run tests including slow tests
test-slow:
    uv run pytest tests/ -v --run-slow

# Run tests including GPU tests (requires compatible GPU)
test-gpu:
    uv run pytest tests/ -v --run-gpu

# Run only smoke tests (fastest feedback loop)
test-smoke:
    uv run pytest tests/ -v -m smoke

# Run specific test module
test-module module:
    uv run pytest tests/test_{{module}}.py -v

# Run benchmark suite
benchmark:
    uv run pytest tests/test_benchmark.py -v -s --run-slow

# Run Rust unit tests
test-rust:
    cargo test --release

# Run everything (CI equivalent)
test-all: test-rust test-slow
    @echo "✅ All tests passed"

# CI workflow (used in GitHub Actions)
ci: pre-commit test-rust test
    @echo "✅ CI checks passed!"

# Full release workflow
release: fmt lint test-all build wheel
    @echo "✅ Release checks passed!"
```

### GitHub Actions CI matrix

The existing `.github/workflows/test.yml` is updated to:

1. Add a `smoke` job that runs `just test-smoke` — fast (< 30 s) feedback on every push.
2. Rename the existing `test` job to `test-full` — runs the full matrix (3 OS × 4 Python versions).
3. Add an optional `test-gpu` job with `workflow_dispatch` trigger only.
4. Ensure all jobs use `uv run pytest` (not bare `pytest`) for reproducible environments.
5. Use the `just` task runner in CI:

```yaml
# .github/workflows/test.yml (excerpt)
- name: Run tests
  run: just test
```

### Benchmark conventions

- All benchmark tests live in `tests/test_benchmark.py`.
- Benchmark tests are marked `@pytest.mark.slow` and `@pytest.mark.benchmark`.
- Benchmarks print tabular output using `print()` (captured with `-s`).
- Benchmark thresholds (e.g., "must complete in < 10 s") are asserted as soft warnings, not hard failures, to avoid flakiness.

### Coverage

Use `pytest-cov` to measure coverage:

```just
# Measure test coverage
coverage:
    uv run pytest tests/ --cov=pcl_rustic --cov-report=html --cov-report=term-missing
```

Target: ≥ 80% line coverage on the Python `pcl_rustic` package.

### Context7 for API documentation

When implementing or reviewing tests, use [context7](https://context7.com) to look up the most current API docs for:

- `pyo3` — for correct Python↔Rust type mappings
- `burn` — for tensor operation signatures
- `kiddo` — for k-d tree query API
- `las` — for LAS/LAZ file format details

This ensures tests use current, not deprecated, API surfaces.

## Drawbacks

- Adding `pytest-cov` and fixture files slightly increases the test setup time.
- The layered marker system requires contributors to choose the right marker; poorly marked tests will slip into the wrong bucket.
- Maintaining pre-generated fixture files (`.npy`) risks version skew if the file format changes.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| Makefile | Already using `just`; Makefile is harder to read and not cross-platform |
| tox | Adds configuration overhead; `uv` already handles virtual environments |
| hypothesis (property-based) | Useful but adds complexity; can be layered on top later |
| cargo-nextest | Faster Rust test runner; can replace `cargo test` in a future RFC |

## Unresolved Questions

1. Should we add `mypy` or `pyright` type-checking as a CI step? (Stub files already exist.)
2. Should `pytest-cov` be a hard CI requirement (fail if coverage drops below threshold)?
3. What is the right session-scope strategy for expensive fixtures (`medium_cloud`) on Windows (slower I/O)?
4. Should we add `mutmut` (mutation testing) to catch weak tests?

## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — establishes pytest, justfile, and uv conventions formalised here
- [RFC-0002 — GPU Acceleration](./RFC-0002-gpu-acceleration.md) — defines the `gpu` marker consumed here
- [RFC-0003 — Advanced Downsampling](./RFC-0003-advanced-downsampling.md) — `test_downsampling.py` test file
- [RFC-0004 — Point Cloud Registration](./RFC-0004-point-cloud-registration.md) — `test_registration.py` and `slow` marker
- [RFC-0005 — Normal Estimation](./RFC-0005-normal-estimation.md) — `test_normals.py` test file
- [RFC-0006 — Point Cloud Segmentation](./RFC-0006-point-cloud-segmentation.md) — `test_segmentation.py` and fixture requirements

## Reviews

### Review 1 — Sub-agent Alpha

> **Status**: APPROVED
>
> This RFC is well-structured and fills a real gap. The marker system (`slow`, `gpu`, `benchmark`, `integration`, `smoke`) is clear and actionable. The `conftest.py` design with `pytest_collection_modifyitems` is the correct idiomatic way to skip markers. The fixture scope (`session`) for large clouds is appropriate. One recommendation: the `smoke` marker should be applied to 1-2 tests per module to give maximum coverage in minimum time — document this expectation in the RFC README.

### Review 2 — Sub-agent Beta

> **Status**: APPROVED (with minor comments)
>
> The justfile recipe additions are comprehensive and cross-platform-safe. The GitHub Actions update (adding a smoke job) will significantly improve CI feedback time — this is the most impactful change in this RFC. The context7 recommendation is excellent practice. One concern: the `fixtures/` subdirectory with `.npy` files may bloat the repository if the fixture sizes grow. Consider generating fixtures at test time using `conftest.py` session-scoped fixtures (already proposed above) instead of committing binary files. Unresolved Question 1 (mypy/pyright) should be addressed soon — the stub files already exist.
