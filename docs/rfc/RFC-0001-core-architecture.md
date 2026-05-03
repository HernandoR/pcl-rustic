# RFC-0001: Core Architecture

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-01-31 |
| **Updated** | 2026-01-31 |
| **Related RFCs** | [RFC-0002](RFC-0002-enhanced-io.md) · [RFC-0003](RFC-0003-advanced-downsampling.md) · [RFC-0004](RFC-0004-point-cloud-registration.md) · [RFC-0005](RFC-0005-normal-estimation.md) · [RFC-0006](RFC-0006-gpu-acceleration.md) · [RFC-0007](RFC-0007-testing-infrastructure.md) |

## Summary

Establish the foundational architecture for PCL Rustic: a high-performance Python point-cloud processing library built on **Rust + PyO3 + Burn**, exposing a single, NumPy-friendly `PointCloud` Python class while keeping all heavy computation in safe, zero-copy Rust code.

## Motivation

Existing Python point-cloud libraries (Open3D, PyPCD, laspy) either lack GPU-acceleration paths, rely on C++ extensions that are difficult to compile on diverse platforms, or have no type-safe Python stubs. PCL Rustic aims to provide:

1. **Native performance** via Rust's ownership model and SIMD/vectorized execution.
2. **GPU-readiness** through the Burn tensor framework (supports NdArray, Tch, Wgpu backends).
3. **Zero-friction Python** interop: NumPy `float32` arrays in, NumPy arrays out, with complete `.pyi` type stubs.
4. **Sustainable engineering** through Trait-based abstraction, making it easy to add new algorithms without touching core data structures.

## Design

### Data Model

```rust
pub struct HighPerformancePointCloud {
    xyz:        Vec<Vec<f32>>,                    // [M, 3], required
    intensity:  Option<Vec<f32>>,                 // [M],    optional
    rgb:        Option<Vec<Vec<u8>>>,             // [M, 3], optional
    attributes: HashMap<String, Vec<f32>>,        // dynamic
}
```

All fields are **private**; access is via Trait methods or `#[pymethods]`.

### Trait Abstraction

| Trait | Methods | Purpose |
|-------|---------|---------|
| `PointCloudCore` | `point_count`, `get_xyz`, `set_xyz`, `clone_cloud`, `is_empty`, `merge` | Core data access |
| `PointCloudProperties` | `has_intensity`, `get_intensity`, `set_intensity`, `get_rgb`, `set_rgb` | Optional attributes |
| `CoordinateTransform` | `transform`, `rigid_transform` | Affine/rigid transforms |
| `VoxelDownsample` | `voxel_downsample` | Spatial downsampling |
| `DownsampleStrategy` | `select_representative`, `name` | Strategy pattern |
| `IOConvert` | `from_las`, `to_las`, `from_csv`, `to_csv`, `from_parquet`, `to_parquet`, `delete_file` | Serialization |

### Python Binding Layer

`src/lib.rs` wraps `HighPerformancePointCloud` in `PyPointCloud` using `#[pyclass]` / `#[pymethods]`. The Python user sees one class (`PointCloud`) and one enum-like class (`DownsampleStrategy`).

```python
from pcl_rustic import PointCloud, DownsampleStrategy
pc = PointCloud.from_xyz(np.random.randn(10_000, 3).astype(np.float32))
pc_down = pc.voxel_downsample(0.15, DownsampleStrategy.CENTROID)
```

### Module Layout

```
src/
├── lib.rs             # PyO3 bindings entry point
├── traits/            # Trait interfaces
├── point_cloud/       # Core struct + algorithm impls
├── io/                # Multi-format I/O
├── utils/             # Error types, tensor validation, voxel grouping
└── interop/           # NumPy array conversion
```

### Build & Distribution

- **Build tool**: `maturin` (PEP 517 build backend)
- **Package manager**: `uv` (replaces pip/poetry in all workflows)
- **Task runner**: `just` (replaces Make)
- **Wheel targets**: manylinux, musllinux, macOS (x86_64 + arm64), Windows (x86 + x86_64)

### Alternatives Considered

| Option | Reason Not Chosen |
|--------|-------------------|
| C++ extension (pybind11) | Compile complexity, no memory safety guarantees |
| Pure Python + NumPy | Insufficient performance for 10M+ point clouds |
| ctypes / cffi | Poor ergonomics, no type safety |
| CFFI with libpcl | Heavy C++ dependency, hard cross-compilation |

## Impact

### Breaking Changes

N/A (initial design).

### Performance Implications

- 10M-point voxel downsampling: ~7 s on Apple M1 (CPU, NdArray backend)
- Throughput: 1.3–1.5 M pts/s

### Testing Plan

- 40+ pytest test cases in `tests/test_point_cloud.py` covering lifecycle, attributes, transform, downsampling, I/O, edge cases, and integration.
- Rust unit tests via `cargo test`.
- CI via GitHub Actions (Ubuntu / macOS / Windows × Python 3.10–3.13).

## Unresolved Questions

- GPU backend selection strategy (see [RFC-0006](RFC-0006-gpu-acceleration.md)).
- Normal estimation algorithm (see [RFC-0005](RFC-0005-normal-estimation.md)).
- Point cloud registration API (see [RFC-0004](RFC-0004-point-cloud-registration.md)).

## References

- [PyO3 User Guide](https://pyo3.rs/)
- [Burn Framework](https://burn.dev/)
- [Maturin Documentation](https://maturin.rs/)
- [just task runner](https://just.systems/)
