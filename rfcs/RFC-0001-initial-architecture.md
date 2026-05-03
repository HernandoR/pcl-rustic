# RFC-0001: Initial Architecture and Design Decisions

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Implemented                        |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2025-01-01                         |
| Updated | 2026-01-31                         |

## Summary

PCL Rustic is a high-performance Python point-cloud processing library built on top of Rust, [PyO3](https://pyo3.rs/), and the [Burn](https://github.com/tracel-ai/burn) tensor framework. This RFC documents the foundational architectural decisions that were made during the initial design of the library, providing background knowledge for all subsequent RFCs.

## Motivation

Existing Python point-cloud libraries (Open3D, python-pcl, PyLidar3D) offer either Python-native implementations that are slow at scale, or C++ bindings that are difficult to install and extend. PCL Rustic aims to provide:

1. **High throughput**: 1.3–1.5 M pts/s voxel downsampling on CPU.
2. **Easy installation**: Pure-Rust backend compiled to a single wheel; no system C++ dependencies.
3. **Pythonic API**: NumPy arrays as the primary data interchange format.
4. **Future GPU readiness**: Burn's backend abstraction allows swapping WGPU/CUDA/Metal without changing Python-facing code.

## Design Details

### Technology Stack

| Layer            | Choice                         | Rationale                                                              |
|------------------|--------------------------------|------------------------------------------------------------------------|
| Core language    | Rust                           | Memory safety, zero-cost abstractions, compile-time guarantees         |
| Python bindings  | PyO3 0.27+                     | Mature, actively maintained, supports `abi3-py39` stable ABI          |
| Tensor compute   | Burn 0.20+                     | Backend-agnostic (ndarray, WGPU, CUDA); trait-based extensibility      |
| Packaging        | Maturin + uv                   | First-class Rust + Python mixed project tooling                        |
| Task runner      | `just` (justfile)              | Simpler than Make; cross-platform; self-documenting                    |
| Testing          | `pytest`                       | Mature Python test framework; CI integration via GitHub Actions         |
| Python linter    | `ruff`                         | Fast, opinionated, replaces flake8 + isort + black                     |
| Rust linter      | `clippy`                       | Official Rust linter; catches common mistakes                          |

### Module Structure

```
src/
├── lib.rs              # PyO3 Python binding entry point
├── traits/             # Trait abstraction layer
│   ├── point_cloud.rs  # Core PointCloud trait
│   ├── io.rs           # I/O interface trait
│   ├── downsample.rs   # Downsample trait
│   └── transform.rs    # Transform trait
├── point_cloud/        # Core data structure
│   ├── core.rs         # HighPerformancePointCloud struct
│   └── voxel.rs        # Voxel downsampling implementation
├── io/                 # Multi-format I/O
│   ├── las_laz.rs      # LAS/LAZ (las + laz-parallel)
│   ├── parquet.rs      # Parquet via arrow/polars
│   └── csv.rs          # CSV via csv crate
├── interop/            # Python interoperability
│   └── numpy.rs        # NumPy array zero-copy read
└── utils/
    ├── error.rs        # Error types (thiserror)
    └── tensor.rs       # Burn tensor helpers
```

### Core Design Constraints

1. **dtype = float32 only**: All input NumPy arrays must be `float32`. Users are responsible for casting. This avoids runtime branching and simplifies the Burn tensor pipeline.
2. **Batch operations only**: No single-point access methods. All operations work on the full point set.
3. **NumPy interface**: Getters return `numpy.ndarray`; setters accept `numpy.ndarray`. Zero-copy read is supported where Rust allows it.
4. **Immutable construction**: `PointCloud` is built from arrays; no incremental insertion API.

### Supported File Formats (v0.1)

| Format  | Read | Write | Notes                               |
|---------|------|-------|-------------------------------------|
| LAS     | ✅   | ✅    | Via `las` crate                     |
| LAZ     | ✅   | ✅    | Via `laz-parallel` feature          |
| CSV     | ✅   | ✅    | Configurable delimiter              |
| Parquet | ⚠️   | ⚠️    | Scaffold present; not fully exposed |

### Downsampling Strategies (v0.1)

| Strategy   | Description                                          |
|------------|------------------------------------------------------|
| `RANDOM`   | One random point per voxel                           |
| `CENTROID` | Point nearest to the voxel centroid is selected      |

### Performance Baseline (MacBook M1, CPU)

| Input pts | Voxel | Output pts | Reduction | Time  | Throughput |
|-----------|-------|------------|-----------|-------|------------|
| 10 M      | 0.06  | 8.8 M      | 11.6 %    | 7.7 s | 1.3 M/s   |
| 10 M      | 0.20  | 7.0 M      | 29.5 %    | 6.4 s | 1.5 M/s   |
| 50 M      | 0.06  | 41.7 M     | 16.5 %    | 47 s  | 1.1 M/s   |

## Alternatives Considered

### Alternative 1: C++ PCL Bindings

Using SWIG or pybind11 to wrap the C++ PCL library was evaluated but rejected because:
- Heavy C++ dependency chain (VTK, Boost, etc.) makes packaging difficult.
- ABI instability across platforms and compilers.
- Slower iteration; safer changes are harder.

### Alternative 2: Pure Python (NumPy/SciPy)

- Insufficient throughput for production-scale point clouds (50 M+ pts).
- Would require Numba JIT as a fallback, adding complexity.

### Alternative 3: Polars-only Backend (no Burn)

- Polars is excellent for tabular data but lacks tensor/spatial computation primitives.
- Would require reinventing voxelization and spatial indexing from scratch.

## Open Questions at Time of Writing

*(All resolved; listed for historical context.)*

1. **GPU backend selection**: WGPU was chosen for cross-platform GPU support. CUDA support to be added later via a Burn backend feature flag. → See [RFC-0002](./RFC-0002-gpu-acceleration.md).
2. **More downsampling strategies**: FPS and normal-based were deferred. → See [RFC-0003](./RFC-0003-downsampling-strategies.md).
3. **Parquet I/O completeness**: Parquet scaffolding is present; full exposure deferred. → See [RFC-0007](./RFC-0007-parquet-format.md).

## Implementation Plan

All items below were completed in the initial release (v0.1.0):

- [x] Core `PointCloud` struct and PyO3 bindings
- [x] NumPy interop (zero-copy read, owned-copy write)
- [x] Voxel downsampling (RANDOM + CENTROID)
- [x] LAS/LAZ file I/O
- [x] CSV file I/O
- [x] Rigid transform and 4×4 matrix transform
- [x] Custom attribute storage (HashMap<String, Vec<f32>>)
- [x] CI: multi-platform GitHub Actions (Ubuntu / macOS / Windows, Python 3.10–3.13)
- [x] `justfile` task runner
- [x] `pytest` integration tests

## Related RFCs

- **Informs** → [RFC-0002](./RFC-0002-gpu-acceleration.md): GPU acceleration
- **Informs** → [RFC-0003](./RFC-0003-downsampling-strategies.md): Additional downsampling strategies
- **Informs** → [RFC-0004](./RFC-0004-point-cloud-registration.md): Point cloud registration
- **Informs** → [RFC-0005](./RFC-0005-normal-estimation.md): Normal vector estimation
- **Informs** → [RFC-0006](./RFC-0006-segmentation.md): Point cloud segmentation
- **Informs** → [RFC-0007](./RFC-0007-parquet-format.md): Parquet format I/O

## References

- [Burn Framework](https://github.com/tracel-ai/burn)
- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [las crate](https://docs.rs/las/)
- [polars crate](https://docs.rs/polars/)
