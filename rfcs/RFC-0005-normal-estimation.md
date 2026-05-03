# RFC-0005: Normal Vector Estimation

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Draft                              |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2026-05-03                         |
| Updated | 2026-05-03                         |

## Summary

Add surface normal estimation to PCL Rustic. Normals are unit vectors perpendicular to the local surface at each point, computed from the Principal Component Analysis (PCA) of k-nearest neighbors. Normals are a prerequisite for Point-to-Plane ICP ([RFC-0004](./RFC-0004-point-cloud-registration.md)), Normal-based downsampling ([RFC-0003](./RFC-0003-downsampling-strategies.md)), and segmentation ([RFC-0006](./RFC-0006-segmentation.md)).

## Motivation

Surface normals encode local geometry that is invisible from raw XYZ coordinates alone. They are required for:

1. **Point-to-Plane ICP** ([RFC-0004](./RFC-0004-point-cloud-registration.md)): normal of the target surface is used in the error metric.
2. **Normal-based downsampling** ([RFC-0003](./RFC-0003-downsampling-strategies.md)): select points with highest local curvature.
3. **Region Growing Segmentation** ([RFC-0006](./RFC-0006-segmentation.md)): cluster points whose normals are similar.
4. **Surface Reconstruction**: Poisson reconstruction (future RFC) requires oriented normals.
5. **Feature Descriptors**: FPFH descriptors (future RFC-0010) use normals.

## Design Details

### Python API

```python
# Compute normals using k-NN (stored in the PointCloud's attributes)
pc.compute_normals(k=20)

# Or with radius search
pc.compute_normals(radius=0.1)

# Access normals
normals = pc.get_normals()   # np.ndarray shape (N, 3), dtype=float32, unit vectors

# Remove normals
pc.remove_normals()

# Check if normals are available
has_normals: bool = pc.has_normals()

# Orient normals consistently toward a viewpoint
pc.orient_normals_toward_viewpoint(viewpoint=np.array([0.0, 0.0, 1.0], dtype=np.float32))
```

### Algorithm: PCA-based Normal Estimation

For each point p_i:
1. Find k nearest neighbors (or all points within radius r).
2. Compute the centroid of the neighborhood.
3. Build the 3×3 covariance matrix of centered neighbor positions.
4. Compute eigenvalues λ_0 ≤ λ_1 ≤ λ_2 and eigenvectors.
5. The normal is the eigenvector corresponding to the **smallest** eigenvalue λ_0 (direction of least variance = perpendicular to local surface).
6. **Curvature estimate** = λ_0 / (λ_0 + λ_1 + λ_2).

### Normal Orientation (Sign Disambiguation)

The PCA eigenvector has an arbitrary sign. Disambiguation strategies:

| Strategy               | Method                                          | Default |
|------------------------|-------------------------------------------------|---------|
| `Viewpoint`            | Flip normal so it faces toward a camera/sensor  | ✅ Yes  |
| `ConsistentPropagation`| Propagate orientation through a MST             | ❌ No   |
| `None`                 | No orientation (raw PCA output)                 | ❌ No   |

Default: viewpoint-based orientation with viewpoint `(0, 0, 0)` (sensor at origin).

### Curvature Storage

Curvature values are stored as a named attribute `"curvature"` (dtype=float32, shape (N,)):

```python
curvature = pc.get_attribute("curvature")  # shape (N,), float32
```

### KNN Backend

Same `KnnBackend` trait as defined in [RFC-0004](./RFC-0004-point-cloud-registration.md):
- CPU: KD-Tree (`kiddo` crate).
- GPU: Burn tensor brute-force kNN ([RFC-0002](./RFC-0002-gpu-acceleration.md)).

### Parallelism

Normal estimation for N points with k neighbors is embarrassingly parallel. CPU path uses `rayon`; GPU path uses Burn tensor batch operations.

### Data Validation

- Normals are stored normalized (unit length). An assertion verifies |normal| ≈ 1 with tolerance 1e-5.
- If fewer than 3 neighbors exist for a point, normal = (0, 0, 1) with curvature = 0 and a warning is emitted.

### Module Layout

```
src/
└── geometry/
    ├── mod.rs          # module re-exports
    ├── normals.rs      # PCA normal estimation
    └── curvature.rs    # curvature helpers (may be merged with normals.rs)
```

Normals are stored in `HighPerformancePointCloud` as an `Option<Vec<[f32; 3]>>` field, separate from the general `HashMap<String, Vec<f32>>` attribute store, for efficient access.

## Alternatives Considered

### Alternative 1: Store Normals as Generic Attribute

Store normals as three separate `f32` attributes (`normal_x`, `normal_y`, `normal_z`) in the existing attribute HashMap. Rejected because:
- Access requires three HashMap lookups per operation.
- No way to enforce unit-norm invariant at the type level.
- Interop with algorithms in [RFC-0004](./RFC-0004-point-cloud-registration.md) and [RFC-0006](./RFC-0006-segmentation.md) is cumbersome.

### Alternative 2: Open3D-style NaN for Missing Normals

Store NaN for points with insufficient neighbors instead of defaulting to (0,0,1). Rejected because:
- NaN propagation in Burn tensors is backend-dependent and unreliable.
- Explicit handling with a warning is safer.

## Open Questions

1. **K vs radius search**: Should the API allow both simultaneously (e.g., "at most k neighbors within radius r")? Open3D supports this; it adds complexity.
2. **Eigenvalue decomposition in Burn**: Burn does not currently expose a GPU SVD/eigen primitive for small (3×3) matrices. A custom Jacobi iteration may be required.
3. **Consistent normal orientation** (MST-based): Needed for surface reconstruction. Deferred to the surface reconstruction RFC, or added here?
4. **Normals in file I/O**: LAS does not have a standard normal field. Should normals be stored in the `extra bytes` LAS extension or only in Parquet ([RFC-0007](./RFC-0007-parquet-format.md))?

## Implementation Plan

- [ ] Add `normals: Option<Vec<[f32; 3]>>` field to `HighPerformancePointCloud`
- [ ] Implement KD-Tree kNN search in `src/geometry/knn.rs` (shared with RFC-0004)
- [ ] Implement PCA normal estimation in `src/geometry/normals.rs`
- [ ] Implement viewpoint-based orientation in `src/geometry/normals.rs`
- [ ] Expose `compute_normals()`, `get_normals()`, `has_normals()`, `remove_normals()`, `orient_normals_toward_viewpoint()` via PyO3
- [ ] Add pytest tests: `tests/test_normals.py`
  - `test_normals_planar_cloud()` — flat XY plane → normals should be ≈ (0,0,±1)
  - `test_normals_unit_length()` — all normals have magnitude ≈ 1.0
  - `test_normals_curvature_stored()` — curvature attribute present after compute
  - `test_normals_requires_min_neighbors()` — <3 neighbors emits warning and uses default
  - `test_orient_normals_toward_viewpoint()` — all normals dot product with viewpoint > 0
  - `test_normals_radius_search()` — radius-based search returns correct normals
- [ ] Update `rfcs/README.md` index status

## Related RFCs

- **Background**: [RFC-0001](./RFC-0001-initial-architecture.md) — attribute storage design
- **Required by** → [RFC-0003](./RFC-0003-downsampling-strategies.md): NormalBased downsampling strategy
- **Required by** → [RFC-0004](./RFC-0004-point-cloud-registration.md): Point-to-Plane ICP
- **Required by** → [RFC-0006](./RFC-0006-segmentation.md): region growing segmentation
- **Accelerated by** ← [RFC-0002](./RFC-0002-gpu-acceleration.md): GPU batch kNN and PCA
- **Relevant for** → [RFC-0007](./RFC-0007-parquet-format.md): normal storage in Parquet schema

## References

- Hoppe et al. "Surface Reconstruction from Unorganized Points" (SIGGRAPH 1992) — PCA normals
- Rusu et al. "Towards 3D Point Cloud Based Object Maps for Household Environments" (2008) — curvature
- [PCL Normal Estimation tutorial](https://pointclouds.org/documentation/tutorials/normal_estimation.html)
- [Open3D estimate_normals](http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.estimate_normals)
- [kiddo Rust crate](https://docs.rs/kiddo/)
- [RFC-0001](./RFC-0001-initial-architecture.md)
- [RFC-0002](./RFC-0002-gpu-acceleration.md)
- [RFC-0003](./RFC-0003-downsampling-strategies.md)
- [RFC-0004](./RFC-0004-point-cloud-registration.md)
- [RFC-0006](./RFC-0006-segmentation.md)
