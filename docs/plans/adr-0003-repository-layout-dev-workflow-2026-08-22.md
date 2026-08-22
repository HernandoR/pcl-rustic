# ADR-0003: Repository Layout & Dev Workflow

- Status: Accepted
- Date: 2026-08-22
- Provenance: RFC-0001 §1 (items 1–3)

## Context

Rust code and the Python package shared `src/`; a `traits/` layer abstracted
a single implementation (`HighPerformancePointCloud` was the only impl of
every trait); voxel logic was smeared across three modules; RFCs and ADRs
shared `docs/plans/`; and the Python env was broken (`pyproject.toml` readme
pointed at a deleted file, failing every `uv run`) and noisy (forced
coverage/junit, benchmark invocation by fully qualified test path).

## Decision

### Source layout

```
src/                      # Rust only
  lib.rs                  # #[pymodule] _core — registration only
  error.rs                # Error enum + PyErr mapping
  backend.rs              # Dispatch device selection & probing (ADR-0001)
  cloud/
    mod.rs                # PointCloud struct — the single mutation choke point
    dims.rs               # standard-dimension registry, DimArray column store (ADR-0002)
    transform.rs          # matrix / rigid transforms
    voxel.rs              # voxel grouping + downsample strategies (enum, not trait objects)
  io/
    mod.rs                # extension-based format detection, read/write entry
    las.rs                # LAS/LAZ via `las` crate
    csv.rs                # CSV via `csv` crate
    parquet.rs            # Parquet via `arrow`/`parquet` crates (polars dropped)
  py/
    mod.rs
    cloud.rs              # #[pyclass] PointCloud bindings
    convert.rs            # numpy / DLPack boundary (ADR-0002 coercion rules)
python/pcl_rustic/        # Python package (maturin python-source = "python")
  __init__.py             # torch/numpy boundary sugar, attribute access, re-exports
  _core.pyi
  py.typed
tests/                    # pytest; markers: slow, bench
docs/rfc/                 # append-only RFCs + index.md
docs/plans/               # ADRs + index.md
```

The former `traits/` layer is deleted: with one implementation, traits added
indirection without polymorphism. Downsample strategies become a plain Rust
enum (replacing the former `Box<dyn DownsampleStrategy>`): variants carry
their parameters (`Random { seed }`, `NearestToCentroid`), and the
middle-index pseudo-"random" bug dies with the old code.

### Dev workflow

- `pyproject.toml`: `readme = "README.md"`; runtime deps = numpy only;
  pytest config lives in `[tool.pytest.ini_options]` (quiet by default, no
  forced coverage); `pytest.ini` deleted.
- Test tiers via markers, driven by `just`:
  `just test` → `pytest -m "not slow and not bench"` (after `maturin develop`);
  `just test-all` → everything except `bench`; `just bench` → `-m bench`.
- Coverage and junit output are CI concerns, passed as flags in the workflow,
  never defaults.

## Consequences

**Positive:** `uv run pytest` works from a fresh clone with zero setup beyond
`just dev`; Rust reviewers never scroll past `.py`; each concern has exactly
one module; the RFC/ADR conventions are now enforceable per folder.

**Negative:** moving the Python package breaks any tooling that hard-coded
`src/pcl_rustic`; deleting traits means a future second implementation (e.g.
a GPU-resident column store) must reintroduce an interface — acceptable, as
speculative interfaces were exactly the problem.
