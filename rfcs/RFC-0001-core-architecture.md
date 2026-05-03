# RFC-0001 — Core Architecture & Design Decisions

| Field | Value |
|-------|-------|
| RFC | 0001 |
| Status | Implemented |
| Author | liuzhen19 |
| Created | 2026-01-31 |
| Updated | 2026-05-03 |

## Summary

This RFC documents the foundational architecture and design decisions for **PCL Rustic** — a high-performance Python point cloud processing library built on Rust, PyO3, and the Burn tensor framework. It serves as the canonical reference for all subsequent RFCs.

## Motivation

Python is the dominant language in the geospatial and robotics communities, but pure-Python point cloud processing at scale suffers from the GIL, interpreter overhead, and lack of vectorised bulk operations. Existing libraries (Open3D, python-pcl) either lack flexibility or are hard to extend with custom algorithms.

PCL Rustic addresses these gaps by combining:

- **Rust** for memory-safe, zero-cost abstractions and native SIMD.
- **PyO3** for ergonomic, ABI-stable Python bindings.
- **Burn** as a portable tensor backend supporting both CPU and GPU execution graphs.

## Design

### Language & build stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Core computation | Rust 1.70+ | Memory safety, SIMD, no GC pauses |
| Python bindings | PyO3 `^0.27` | ABI3 stable, async-safe, low overhead |
| Tensor framework | Burn `0.20.1` | Supports ndarray (CPU) and wgpu (GPU) backends via feature flags |
| Build tool | maturin + uv | Cross-platform wheel generation, reproducible lock files |
| Task runner | `just` (justfile) | Replaces Makefile; simple, cross-platform |
| Python tests | pytest | Industry standard; supports markers and CI integration |

### Module hierarchy

```
src/
├── lib.rs              # PyO3 module root; registers Python classes/enums
├── traits/             # Trait abstraction layer (public interface contracts)
│   ├── point_cloud.rs  # PointCloudCore, PointCloudProperties
│   ├── io.rs           # IOConvert
│   ├── downsample.rs   # VoxelDownsample, DownsampleStrategy
│   └── transform.rs    # CoordinateTransform
├── point_cloud/        # Concrete implementations
│   ├── core.rs         # HighPerformancePointCloud struct
│   ├── attributes.rs   # Attribute CRUD (HashMap<String, Vec<f32>>)
│   ├── transform.rs    # 3×3 / 4×4 matrix and rigid-body transforms
│   └── voxel.rs        # Voxel grid downsampling + strategy dispatch
├── io/
│   ├── las_laz.rs      # LAS / LAZ read-write (las crate)
│   ├── parquet.rs      # Apache Parquet (arrow + parquet crates)
│   └── csv.rs          # CSV (csv crate)
├── interop/
│   └── numpy.rs        # Zero-copy NumPy ↔ Rust Vec<f32> bridging
└── utils/
    ├── error.rs        # PointCloudError + From<PointCloudError> for PyErr
    ├── tensor.rs       # Burn tensor helpers & dtype validation
    └── reflect.rs      # Voxel grouping (O(N) hash-based)
```

### Core data structure

```rust
pub struct HighPerformancePointCloud {
    xyz:        Vec<Vec<f32>>,              // [N, 3] — always present
    intensity:  Option<Vec<f32>>,           // [N]    — optional
    rgb:        Option<Vec<Vec<u8>>>,       // [N, 3] — optional
    attributes: HashMap<String, Vec<f32>>, // arbitrary named channels
}
```

All fields are private. Access is exclusively through trait methods to enforce batch-only semantics (no single-point get/set).

### Trait hierarchy

```
PointCloudCore          point_count, get_xyz, has_intensity, …
PointCloudProperties    set_intensity, add_attribute, remove_attribute, …
CoordinateTransform     transform(4×4), rigid_transform(R, t)
VoxelDownsample         voxel_downsample(size, strategy)
DownsampleStrategy      select_representative(indices, xyz) → usize
IOConvert               from_las, from_csv, from_parquet, to_*, delete_file
```

### Python API surface

Exposed via a single Python class `PointCloud` (backed by `PyPointCloud` in Rust), plus a `DownsampleStrategy` enum with `RANDOM` and `CENTROID` variants.

```python
pc = PointCloud.from_xyz(xyz_np_f32)   # dtype must be float32
pc.set_intensity(intensity_np_f32)
pc_down = pc.voxel_downsample(voxel_size=0.1, strategy=DownsampleStrategy.CENTROID)
pc.to_las("out.las", compress=True)
```

### Error handling

`PointCloudError` (thiserror) is the single error type for all Rust failures. A `From<PointCloudError> for PyErr` impl maps each variant to the most appropriate Python built-in exception:

| Rust variant | Python exception |
|---|---|
| `DimensionMismatch` | `ValueError` |
| `FileNotFound` | `FileNotFoundError` |
| `IoError` | `IOError` |
| `InvalidDtype` | `TypeError` |

### Performance characteristics

| Operation | Scale | Throughput |
|-----------|-------|-----------|
| Voxel downsample (CENTROID) | 10 M pts | 1.3–1.5 M pts/s |
| Rigid transform | 1 M pts | > 20 M pts/s |
| NumPy round-trip | 10 M pts | < 1 s |

Memory layout: 12 bytes per point (XYZ f32). Optional channels add 4 bytes each.

## Drawbacks

- Only `float32` is supported for XYZ and attributes. Users must convert before calling library functions.
- Batch-only API is less ergonomic for exploratory/interactive use cases.
- Burn's wgpu backend requires a compatible GPU driver; CI must run on CPU-only paths.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| numpy-only Python | Cannot achieve sustained > 5 M pts/s without GIL contention |
| C++ extension (pybind11) | Manual memory management; slower iteration |
| Cython | Weaker type safety; poor IDE support |
| Apache Arrow / Polars only | No domain-specific point cloud primitives |

## Unresolved Questions

All questions from the initial design phase have been resolved. See RFC-0002 through RFC-0007 for open questions relating to future capabilities.

## Related RFCs

- [RFC-0002 — GPU Acceleration Backend](./RFC-0002-gpu-acceleration.md) — extends the Burn backend selection described here
- [RFC-0003 — Advanced Downsampling](./RFC-0003-advanced-downsampling.md) — adds new `DownsampleStrategy` implementations
- [RFC-0004 — Point Cloud Registration](./RFC-0004-point-cloud-registration.md) — new trait built on top of `CoordinateTransform`
- [RFC-0005 — Normal Estimation](./RFC-0005-normal-estimation.md) — new trait and attribute channel
- [RFC-0006 — Point Cloud Segmentation](./RFC-0006-point-cloud-segmentation.md) — new trait consuming `PointCloudCore`
- [RFC-0007 — Testing & CI Infrastructure](./RFC-0007-testing-ci-infrastructure.md) — formalises the pytest/justfile/CI conventions established here

## Reviews

_N/A — this RFC documents decisions already implemented in the codebase._
