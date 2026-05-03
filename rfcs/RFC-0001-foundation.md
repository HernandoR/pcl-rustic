# RFC-0001: Project Foundation & Architecture

- **Status**: Accepted
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-01-01
- **Updated**: 2026-05-03

---

## Summary

PCL Rustic is a high-performance Python point cloud processing library built on Rust + PyO3 + Burn tensor framework.  
This RFC documents the foundational architecture, key design decisions, and the layered abstraction model that all future RFCs build upon.

---

## Motivation

Python point cloud libraries (Open3D, python-pcl) face performance ceilings due to the GIL and Python's object overhead.  
A Rust core with PyO3 bindings enables near-native throughput while keeping a Python-friendly API.  
Burn's tensor abstraction decouples computation from hardware, enabling transparent CPU/GPU acceleration.

---

## Detailed Design

### Architecture Layers

```
Python API (pcl_rustic package)
        │
        ▼
PyO3 bindings (src/lib.rs)
        │
        ▼
Rust Core (HighPerformancePointCloud)
        │
   ┌────┴──────────────────┐
   ▼                       ▼
Trait Layer              Burn Tensors
(point_cloud, io,        (ndarray / wgpu / NdArray backends)
 downsample, transform)
        │
   ┌────┴────────────────────────┐
   ▼         ▼         ▼        ▼
LAS/LAZ   Parquet    CSV    NumPy interop
```

### Core Struct

```rust
pub struct HighPerformancePointCloud {
    xyz: Vec<Vec<f32>>,                     // [N, 3], mandatory
    intensity: Option<Vec<f32>>,            // [N,], optional
    rgb: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,  // R, G, B channels, optional
    attributes: HashMap<String, Vec<f32>>,  // arbitrary named attributes
}
```

### Trait Abstractions

| Trait | Responsibility |
|-------|---------------|
| `PointCloudCore` | Read-only access: XYZ, intensity, RGB, point count |
| `PointCloudProperties` | Mutable attributes: set/add/remove/get |
| `CoordinateTransform` | Matrix and rigid-body transforms |
| `VoxelDownsample` | Voxel-grid downsampling with pluggable strategy |
| `IOConvert` | Read/write LAS, LAZ, CSV, Parquet |

### Python API Surface

```python
from pcl_rustic import PointCloud, DownsampleStrategy

pc = PointCloud.from_xyz(xyz: np.ndarray)   # dtype=float32, shape=(N,3)
pc.set_intensity(arr: np.ndarray)
pc.rigid_transform(R: np.ndarray, t: np.ndarray) -> PointCloud
pc.voxel_downsample(voxel_size: float, strategy: DownsampleStrategy) -> PointCloud
pc.to_las(path: str, compress: bool = False)
```

### Design Constraints

1. **float32 only** — all NumPy inputs must be `dtype=float32`; the library does not silently cast.
2. **Batch operations only** — no single-point access or mutation; all operations are array-level.
3. **Immutable transforms** — transform/downsample methods return new `PointCloud` objects; they do not mutate in place.
4. **Zero-copy NumPy interop** — XYZ arrays are read directly from NumPy buffers via `rust-numpy`.

### Tooling Decisions

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Build system | `justfile` | Simpler than Makefile, cross-platform |
| Python packaging | `maturin` + `uv` | Best-in-class Rust→Python wheel builder |
| Python testing | `pytest` | Standard; supports markers, fixtures, CI |
| Python linting | `ruff` | Fast, PEP-compliant |
| Rust linting | `clippy` | Official Rust linter |
| CI | GitHub Actions | Matrix test across Ubuntu/macOS/Windows × Python 3.10–3.13 |
| Docs | MkDocs Material | Markdown-based, GitHub Pages deployable |

---

## Implementation Plan

This RFC is _already implemented_ as the baseline of the repository.

- [x] `HighPerformancePointCloud` struct with full attribute support
- [x] PyO3 bindings in `src/lib.rs`
- [x] LAS/LAZ I/O via the `las` crate
- [x] CSV I/O via the `csv` crate
- [x] NumPy interop via `rust-numpy`
- [x] Voxel downsampling (RANDOM, CENTROID strategies)
- [x] pytest test suite in `tests/`
- [x] `justfile` with all developer commands
- [x] GitHub Actions CI

---

## Testing Strategy

All existing tests in `tests/test_point_cloud.py` validate this foundation.  
Run with:

```bash
just test
```

Benchmark tests (slow-marked) run separately:

```bash
just benchmark
```

---

## Alternatives Considered

- **Pure Python + Cython**: rejected; performance ceiling too low for 10M+ point clouds.
- **C++ extension**: rejected; build complexity, no ergonomic tensor framework.
- **JAX/Numba**: rejected; adds large Python dependencies, no zero-copy Rust interop.

---

## Unresolved Questions

- GPU backend selection (WGPU vs CUDA) — addressed in [RFC-0002](./RFC-0002-tensor-backend.md).
- Parquet format full support — addressed in [RFC-0003](./RFC-0003-extended-io.md).

---

## Related RFCs

- [RFC-0002](./RFC-0002-tensor-backend.md) — GPU/tensor backend selection builds on this foundation
- [RFC-0003](./RFC-0003-extended-io.md) — Extended I/O builds on the IOConvert trait defined here
- [RFC-0004](./RFC-0004-sampling-strategies.md) — Advanced sampling extends VoxelDownsample trait
- [RFC-0005](./RFC-0005-registration.md) — Registration algorithms use CoordinateTransform
- [RFC-0006](./RFC-0006-normal-estimation.md) — Normal estimation extends PointCloudCore
- [RFC-0007](./RFC-0007-segmentation.md) — Segmentation extends PointCloudProperties
