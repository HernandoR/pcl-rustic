# RFC-0005: Normal Vector Estimation

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-05-03 |
| **Updated** | 2026-05-03 |
| **Related RFCs** | [RFC-0001](RFC-0001-core-architecture.md) · [RFC-0003](RFC-0003-advanced-downsampling.md) · [RFC-0004](RFC-0004-point-cloud-registration.md) · [RFC-0006](RFC-0006-gpu-acceleration.md) · [RFC-0007](RFC-0007-testing-infrastructure.md) |

## Summary

Add surface-normal estimation to `PointCloud` using **PCA over k-nearest or radius neighbours**. Normals are stored as an optional `[M, 3]` attribute in the existing `attributes` HashMap. Provide orientation normalisation and a fast GPU path (via [RFC-0006](RFC-0006-gpu-acceleration.md)). Normals are prerequisite data for [RFC-0003](RFC-0003-advanced-downsampling.md) (normal-based downsampling) and [RFC-0004](RFC-0004-point-cloud-registration.md) (point-to-plane ICP, FPFH descriptors, RANSAC registration).

## Motivation

Surface normals are ubiquitous in point-cloud processing:

- **Rendering / visualisation**: lighting calculations require per-point normals.
- **Registration**: point-to-plane ICP converges faster than point-to-point.
- **Segmentation**: planar region growing and RANSAC plane fitting need normals.
- **Downsampling**: normal-based voxel weights (RFC-0003).
- **Feature descriptors**: FPFH, SHOT, Spin Images all require normals.
- **Meshing**: Poisson surface reconstruction requires oriented normals.

## Design

### 5.1 Data Storage

Normals are stored as an `[M, 3]` float32 attribute under the reserved key `"__normals__"` in the existing `attributes: HashMap<String, Vec<f32>>`. This avoids a dedicated struct field while remaining accessible via the generic attribute API. A convenience method hides the internal key.

> **Reserved key convention**: attribute keys prefixed with `__` are reserved for internal use and are not serialised by default.

```python
pc.estimate_normals(k=20)              # k-nearest neighbour PCA
pc.estimate_normals(radius=0.1)        # radius-based PCA
has_normals: bool = pc.has_normals()
normals: np.ndarray = pc.get_normals() # shape (N, 3), dtype float32
pc.orient_normals_towards(viewpoint=np.array([0, 0, 0], dtype=np.float32))
pc.remove_normals()
```

### 5.2 PCA-Based Normal Estimation

For each point `p_i`:

1. Find k nearest neighbours `{p_j}` (using the KD-tree from [RFC-0004](RFC-0004-point-cloud-registration.md)).
2. Compute the 3×3 covariance matrix of the neighbourhood.
3. Take the eigenvector corresponding to the **smallest** eigenvalue → surface normal.
4. Curvature = `λ_min / (λ_min + λ_mid + λ_max)` — stored as `"__curvature__"`.

```rust
pub struct NormalEstimator {
    method: NormalMethod,
}

pub enum NormalMethod {
    KNearest(usize),         // k neighbours
    Radius(f32),             // radius search
}

impl NormalEstimator {
    pub fn estimate(
        &self,
        cloud: &HighPerformancePointCloud,
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>)>  // (normals, curvatures)
}
```

Parallelism: each point's neighbourhood is processed independently → `rayon::par_iter`.

### 5.3 Normal Orientation Consistency

Raw PCA normals have sign ambiguity. Two orientation strategies:

**a) Viewpoint orientation** (recommended for single-scan data):

```python
pc.orient_normals_towards(viewpoint=np.array([0.0, 0.0, 5.0], dtype=np.float32))
```

Flip normal `n_i` if `dot(n_i, viewpoint - p_i) < 0`.

**b) Minimum spanning tree (MST) propagation** (for multi-scan or synthetic data):

```python
pc.orient_normals_consistent()   # graph-based consistent orientation
```

Build a graph where vertices are points and edge weights are `1 - |dot(n_i, n_j)|`; propagate sign via Prim's MST.

### 5.4 Rust API

```rust
impl HighPerformancePointCloud {
    pub fn estimate_normals(
        &mut self,
        method: NormalMethod,
    ) -> Result<()>;

    pub fn has_normals(&self) -> bool;

    pub fn get_normals(&self) -> Result<Vec<[f32; 3]>>;

    pub fn orient_normals_towards(
        &mut self,
        viewpoint: [f32; 3],
    ) -> Result<()>;

    pub fn orient_normals_consistent(&mut self) -> Result<()>;

    pub fn remove_normals(&mut self);
}
```

### 5.5 Dependency Addition

| Crate | Purpose |
|-------|---------|
| `nalgebra` 0.33 | 3×3 covariance matrix + eigen-decomposition |
| `kiddo` (shared with RFC-0004) | KD-tree neighbour queries |

### Alternatives Considered

| Approach | Reason Not Chosen |
|----------|-------------------|
| Dedicated `normals` field in struct | Breaks backward compat; `attributes` map is sufficient |
| Python-side PCA (scikit-learn) | 100× slower; defeats the purpose of Rust core |
| Store normals as `rgb`-style `Option<Vec<Vec<f32>>>` | Inconsistent with attribute API |

## Impact

### Breaking Changes

None. Normals are stored in the existing `attributes` map under a reserved key; existing attribute keys are unaffected.

### Performance Implications

| Operation | Expected time (1 M pts, k=20) |
|-----------|-------------------------------|
| KD-tree build | ~200 ms |
| PCA estimation (CPU, rayon) | ~800 ms |
| Viewpoint orientation | ~50 ms |
| MST orientation | ~500 ms |

GPU acceleration path deferred to [RFC-0006](RFC-0006-gpu-acceleration.md).

### Testing Plan

- `tests/test_normals.py`:
  - Normals estimated for a flat XY-plane point cloud should all point in the ±Z direction.
  - Normals for a sphere should point radially outward after viewpoint orientation.
  - `has_normals()` returns `False` before and `True` after `estimate_normals()`.
  - `remove_normals()` clears the attribute.
  - Curvatures are in `[0, 1]` range.
  - Edge cases: cloud with < k points, single-point cloud.
- Integration test: `estimate_normals` output feeds into `voxel_downsample` with `DownsampleStrategy.NORMAL` (requires RFC-0003).
- API docs: use context7 for `nalgebra` eigendecomposition and `kiddo` radius search APIs.

## Unresolved Questions

1. Should `__curvature__` be computed automatically alongside normals, or require a separate call?
2. For radius search with non-uniform density, should we cap the maximum number of neighbours?
3. Should the MST orientation be parallelised with rayon (requires a concurrent MST algorithm)?
4. How should normals be serialised to Parquet and LAS? (LAS 1.4 has no standard normal field; custom VLR?)

## References

- Hoppe et al. (1992) "Surface reconstruction from unorganized points" — PCA normal estimation.
- Lalonde et al. (2005) "Scale selection for classification of point-sampled 3D surfaces" — curvature.
- [nalgebra crate](https://nalgebra.org/)
- [RFC-0001: Core Architecture](RFC-0001-core-architecture.md)
- [RFC-0003: Advanced Downsampling](RFC-0003-advanced-downsampling.md)
- [RFC-0004: Point Cloud Registration](RFC-0004-point-cloud-registration.md)
- [RFC-0006: GPU Acceleration Backend](RFC-0006-gpu-acceleration.md)
