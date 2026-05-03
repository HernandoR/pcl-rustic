# RFC-0004: Point Cloud Registration (ICP / NDT)

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Draft                              |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2026-05-03                         |
| Updated | 2026-05-03                         |

## Summary

Add point-cloud registration algorithms to PCL Rustic, starting with **Iterative Closest Point (ICP)** and **Normal Distributions Transform (NDT)**. Registration computes the rigid transformation (rotation + translation) that best aligns a *source* point cloud to a *target* point cloud.

## Motivation

Registration is one of the most fundamental operations in 3D perception pipelines:

- **SLAM** (Simultaneous Localization and Mapping): aligning consecutive LiDAR scans.
- **Map merging**: fusing point clouds from multiple sensors or sessions.
- **Object pose estimation**: aligning a CAD model to a scanned point cloud.

Without registration, PCL Rustic cannot support end-to-end 3D perception workflows. ICP is the canonical baseline; NDT is the industry-preferred algorithm for large-scale outdoor LiDAR (outperforms ICP in speed and noise robustness).

## Design Details

### Python API

```python
from pcl_rustic import RegistrationResult

# Point-to-Point ICP
result: RegistrationResult = pc_source.icp(
    target=pc_target,
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=0.5,
)

# Point-to-Plane ICP (requires normals on target)
result = pc_source.icp(
    target=pc_target,
    point_to_plane=True,  # default: False
    max_iterations=50,
    tolerance=1e-6,
    max_correspondence_distance=0.5,
)

# NDT registration
result = pc_source.ndt(
    target=pc_target,
    resolution=1.0,       # voxel resolution for NDT grid
    max_iterations=35,
    tolerance=0.01,
)

# Result object
print(result.transformation)   # np.ndarray shape (4,4) float32
print(result.fitness)          # float — fraction of inlier correspondences
print(result.inlier_rmse)      # float — RMSE of inlier correspondences
print(result.converged)        # bool
print(result.num_iterations)   # int
```

### RegistrationResult Data Class

```python
@dataclass
class RegistrationResult:
    transformation: np.ndarray  # (4,4) float32
    fitness: float
    inlier_rmse: float
    converged: bool
    num_iterations: int
```

Implemented as a PyO3 `#[pyclass]` with read-only properties.

### ICP Algorithm

**Point-to-Point ICP** (classic):
1. Find nearest-neighbor correspondences in target for each source point.
2. Reject correspondences with distance > `max_correspondence_distance`.
3. Compute least-squares optimal rotation (SVD) and translation.
4. Apply transformation to source.
5. Repeat until convergence or `max_iterations` reached.

**Point-to-Plane ICP** (more accurate on surfaces):
- Minimizes the distance from source points to the tangent plane of the corresponding target points (requires target normals).
- Closed-form solution via linearized rotation (small-angle approximation per iteration).

### NDT Algorithm

Normal Distributions Transform (Biber & Straßer 2003):
1. Divide target into a voxel grid; compute mean `μ_i` and covariance `Σ_i` per voxel.
2. Compute probability of source points under each voxel's Gaussian.
3. Optimize transformation to maximize total probability (Newton's method).

NDT avoids explicit point-to-point correspondences, making it more robust to noise and partial overlap.

### Nearest Neighbor Search

ICP requires efficient kNN. Options:
- **Flat brute-force**: Burn tensor + `argmin` (GPU-friendly, O(N×M) per iteration).
- **KD-Tree**: `kiddo` or `nabo` Rust crate (CPU, faster for large clouds).

Decision: use flat brute-force on GPU ([RFC-0002](./RFC-0002-gpu-acceleration.md)) and KD-Tree on CPU. A `KnnBackend` trait will abstract the choice.

### Spatial Complexity

| Algorithm        | Memory    | Time per iteration |
|------------------|-----------|--------------------|
| Point-to-Point ICP (CPU, KD-Tree) | O(N) | O(N log M) |
| Point-to-Point ICP (GPU, brute)   | O(N + M) | O(N × M / G) |
| NDT (CPU)        | O(V)      | O(N × V_active)    |

Where N = source pts, M = target pts, V = voxels, G = GPU parallelism.

### Module Layout

```
src/
└── registration/
    ├── mod.rs        # module re-exports
    ├── icp.rs        # ICP (point-to-point and point-to-plane)
    ├── ndt.rs        # NDT
    ├── knn.rs        # KNN backend trait + implementations
    └── result.rs     # RegistrationResult struct + PyO3 binding
```

### Initial Value (Initialization)

Both ICP and NDT accept an optional initial transformation matrix to warm-start the optimization:

```python
result = pc_source.icp(
    target=pc_target,
    initial_transform=np.eye(4, dtype=np.float32),  # optional
)
```

## Alternatives Considered

### Alternative 1: Python-side ICP via scipy/sklearn

Reject: too slow for 10 M+ points; lacks GPU path; requires extra dependencies.

### Alternative 2: GICP (Generalized ICP)

GICP (Segal et al.) combines point-to-plane ICP with covariance-aware matching. Recommended for a follow-up RFC (RFC-0009) once basic ICP is stable.

### Alternative 3: Feature-based Registration (FPFH + RANSAC)

More robust to poor initialization but significantly more complex. Requires normal estimation ([RFC-0005](./RFC-0005-normal-estimation.md)) and feature descriptors. Deferred to RFC-0010.

## Open Questions

1. **KD-Tree crate selection**: `kiddo`, `nabo`, or `rstar`? Needs benchmarking for 10 M pts.
2. **Point-to-Plane ICP linearization**: Use exact SVD-based or linearized (small-angle) approach? Linearized is faster per iteration but converges only for small rotations.
3. **NDT voxel grid sharing with downsampling**: Can the NDT voxel grid reuse the same spatial index as voxel downsampling? Would avoid double construction.
4. **Max correspondence distance units**: Should this be in the same coordinate space as the input cloud, or normalized?

## Implementation Plan

- [ ] Create `src/registration/` module structure
- [ ] Implement `KnnBackend` trait with CPU (KD-Tree) implementation
- [ ] Implement Point-to-Point ICP in `src/registration/icp.rs`
- [ ] Implement Point-to-Plane ICP (requires normals; depends on [RFC-0005](./RFC-0005-normal-estimation.md))
- [ ] Implement NDT in `src/registration/ndt.rs`
- [ ] Create `RegistrationResult` PyO3 class
- [ ] Expose `icp()` and `ndt()` methods on `PointCloud` PyO3 class
- [ ] Add pytest tests: `tests/test_registration.py`
  - `test_icp_identity()` — ICP of a cloud with itself returns identity
  - `test_icp_known_transform()` — ICP recovers a known rotation+translation
  - `test_icp_convergence_flag()` — `result.converged` is True for easy case
  - `test_ndt_known_transform()` — NDT recovers a known transform
  - `test_icp_point_to_plane()` — point-to-plane ICP on planar cloud
  - `test_registration_result_fields()` — all result fields have correct types
- [ ] Update `justfile`: no new recipe; covered by `just test`
- [ ] Update `rfcs/README.md` index status

## Related RFCs

- **Background**: [RFC-0001](./RFC-0001-initial-architecture.md) — transform API and dtype conventions
- **Accelerated by** ← [RFC-0002](./RFC-0002-gpu-acceleration.md): GPU nearest-neighbor search
- **Uses** ← [RFC-0003](./RFC-0003-downsampling-strategies.md): FPS for keypoint selection pre-registration
- **Depends on** → [RFC-0005](./RFC-0005-normal-estimation.md): normals required for Point-to-Plane ICP
- **Future** → RFC-0009 (planned): Generalized ICP (GICP)
- **Future** → RFC-0010 (planned): Feature-based registration (FPFH + RANSAC)

## References

- Besl & McKay "A Method for Registration of 3-D Shapes" (IEEE TPAMI 1992) — original ICP
- Low "Linear Least-Squares Optimization for Point-to-Plane ICP Surface Registration" (2004)
- Biber & Straßer "The Normal Distributions Transform: A New Approach to Laser Scan Matching" (IROS 2003)
- [Open3D ICP tutorial](http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
- [kiddo Rust crate](https://docs.rs/kiddo/)
- [RFC-0001](./RFC-0001-initial-architecture.md)
- [RFC-0002](./RFC-0002-gpu-acceleration.md)
- [RFC-0005](./RFC-0005-normal-estimation.md)
