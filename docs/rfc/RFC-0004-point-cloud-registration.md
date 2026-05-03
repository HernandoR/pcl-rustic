# RFC-0004: Point Cloud Registration

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-05-03 |
| **Updated** | 2026-05-03 |
| **Related RFCs** | [RFC-0001](RFC-0001-core-architecture.md) · [RFC-0003](RFC-0003-advanced-downsampling.md) · [RFC-0005](RFC-0005-normal-estimation.md) · [RFC-0006](RFC-0006-gpu-acceleration.md) · [RFC-0007](RFC-0007-testing-infrastructure.md) |

## Summary

Add a point-cloud **registration** subsystem providing **ICP (Iterative Closest Point)** and **NDT (Normal Distribution Transform)** algorithms, plus a **RANSAC-based coarse registration** initialiser. The subsystem returns a `RegistrationResult` containing the converged transformation matrix, fitness score, and RMSE. Both algorithms are exposed as Python methods on `PointCloud`.

## Motivation

Registration — aligning two partially-overlapping point clouds — is the most commonly requested feature missing from the current library. Applications include:

- **LiDAR SLAM**: aligning successive frames from a moving vehicle.
- **3D reconstruction**: stitching multiple scans of a scene.
- **Object localisation**: fitting a known model into a scene scan.
- **Quality control**: comparing a manufactured part against a CAD model.

RFC-0001's Trait-based architecture provides the clean hooks needed (transforms, attribute access) but registration algorithms are not yet implemented.

## Design

### 4.1 `RegistrationResult` Data Class

```python
@dataclass
class RegistrationResult:
    transformation: np.ndarray    # shape (4, 4), dtype float32
    fitness: float                # fraction of inliers (0–1); higher = better
    inlier_rmse: float            # RMSE of inlier correspondences
    converged: bool
    iterations: int
```

Rust-side:

```rust
#[pyclass]
pub struct RegistrationResult {
    pub transformation: [[f32; 4]; 4],
    pub fitness: f32,
    pub inlier_rmse: f32,
    pub converged: bool,
    pub iterations: usize,
}
```

### 4.2 ICP (Point-to-Point and Point-to-Plane)

```python
result = source.icp(
    target,
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_dist=0.05,
    variant="point_to_point",   # or "point_to_plane"
    init_transform=None,        # optional 4×4 ndarray
)
source_aligned = source.transform(result.transformation)
```

Algorithm:

1. For each point in *source*, find nearest neighbour in *target* (KD-tree).
2. Reject correspondences with distance > `max_correspondence_dist`.
3. Solve the least-squares rigid transform (SVD for point-to-point; iterative for point-to-plane).
4. Apply transform; repeat until `|Δtransform| < tolerance` or max iterations reached.

Point-to-plane requires surface normals on *target* (see [RFC-0005](RFC-0005-normal-estimation.md)).

### 4.3 NDT (Normal Distribution Transform)

NDT represents the target cloud as a grid of Gaussian distributions, then optimises the transform that maximises the likelihood of source points falling within those distributions.

```python
result = source.ndt(
    target,
    resolution=1.0,       # voxel size for NDT grid
    max_iterations=30,
    step_size=0.1,
    tolerance=0.01,
    init_transform=None,
)
```

NDT is more robust than ICP when the initial misalignment is large (> 1 m), but requires a resolution parameter roughly equal to the scene scale.

### 4.4 RANSAC Coarse Registration

For the case where no initial estimate is available, a feature-based RANSAC pass provides an initial transform:

```python
result = source.ransac_registration(
    target,
    feature="fpfh",          # fast point feature histogram
    num_ransac_iters=100_000,
    inlier_threshold=0.05,
    confidence=0.999,
)
```

FPFH descriptors are computed using normals from [RFC-0005](RFC-0005-normal-estimation.md).

### 4.5 KD-Tree Spatial Index

ICP and RANSAC both require fast nearest-neighbour queries. Introduce an internal `KdTree` structure:

```rust
pub struct KdTree {
    inner: kiddo::KdTree<f32, usize, 3>,
}

impl KdTree {
    pub fn new(cloud: &HighPerformancePointCloud) -> Self { ... }
    pub fn nearest(&self, query: [f32; 3], k: usize) -> Vec<(f32, usize)> { ... }
    pub fn radius(&self, query: [f32; 3], r: f32) -> Vec<(f32, usize)> { ... }
}
```

Crate: `kiddo` (pure Rust, no unsafe, Apache-2.0).

### 4.6 Downsampling Integration

For large clouds (> 1 M points), preprocessing with FPS or voxel downsampling is recommended before registration. Documentation will guide users to [RFC-0003](RFC-0003-advanced-downsampling.md) preprocessing steps.

### Alternatives Considered

| Approach | Reason Not Chosen |
|----------|-------------------|
| Bind to libpcl registration | C++ dependency, breaks cross-compilation |
| Pure Python ICP | 100× slower; not suitable for production pipelines |
| GPU ICP only | Portability requirement; GPU path deferred to RFC-0006 |
| `nanoflann` (C++) KD-tree | C++ FFI surface; `kiddo` is pure Rust |

## Impact

### Breaking Changes

None — all registration methods are additive.

### Performance Implications

| Algorithm | Expected throughput |
|-----------|---------------------|
| ICP (point-to-point, 10k pts) | < 100 ms/iter |
| NDT (1 M pts scene, 100k pts source) | < 500 ms |
| KD-tree build (1 M pts) | < 200 ms |

GPU acceleration path is deferred to [RFC-0006](RFC-0006-gpu-acceleration.md).

### Testing Plan

- `tests/test_registration.py`:
  - ICP converges on synthetic shifted/rotated copy of a source cloud.
  - NDT converges on a noisy scan pair.
  - RANSAC coarse registration returns a reasonable initial transform.
  - `RegistrationResult` fields have correct types and value ranges.
  - Edge cases: empty cloud, clouds with no overlap, single-point cloud.
- Benchmark in `tests/test_benchmark.py`: ICP 10k pts completes < 5 s.
- API docs: use context7 to verify `kiddo` KD-tree API.

## Unresolved Questions

1. Should `KdTree` be exposed as a first-class Python object for user queries (e.g., neighbour search for custom algorithms)?
2. Multi-scale ICP (coarse-to-fine): defer to a follow-up RFC or include here?
3. Should `RegistrationResult.transformation` be a `numpy.ndarray` or a custom `Transform4x4` object?

## References

- Besl & McKay (1992) "A Method for Registration of 3-D Shapes" — original ICP.
- Biber & Strasser (2003) "The Normal Distributions Transform" — NDT.
- [kiddo KD-tree crate](https://docs.rs/kiddo)
- [RFC-0001: Core Architecture](RFC-0001-core-architecture.md)
- [RFC-0003: Advanced Downsampling](RFC-0003-advanced-downsampling.md)
- [RFC-0005: Normal Vector Estimation](RFC-0005-normal-estimation.md)
- [RFC-0006: GPU Acceleration Backend](RFC-0006-gpu-acceleration.md)
