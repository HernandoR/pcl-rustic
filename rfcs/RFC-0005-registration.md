# RFC-0005: Point Cloud Registration (ICP, NDT)

- **Status**: Draft
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-05-03
- **Updated**: 2026-05-03

---

## Summary

Implement point cloud registration algorithms — Iterative Closest Point (ICP) and Normal Distributions Transform (NDT) — in Rust, exposed through a unified `register` Python API that returns a 4×4 transformation matrix and convergence metadata.

---

## Motivation

Point cloud registration is the most frequently requested missing feature.  
It is required for:
- Multi-scan merging (LiDAR SLAM)
- Object pose estimation
- Autonomous driving map construction
- Quality-control inspection (compare design vs. scan)

Existing Python options (Open3D, python-pcl) are heavy dependencies.  
A native Rust implementation avoids the dependency and benefits from the Burn backend (RFC-0002) for parallel distance computation.

---

## Detailed Design

### Unified Registration API

```python
from pcl_rustic import PointCloud, RegistrationMethod, RegistrationResult

result: RegistrationResult = source.register(
    target: PointCloud,
    method: RegistrationMethod = RegistrationMethod.ICP,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_distance: float = 1.0,
    initial_transform: np.ndarray | None = None,  # (4, 4) float32
)

# Result
print(result.transformation)   # np.ndarray (4, 4) float32
print(result.fitness)          # float: overlapping ratio
print(result.inlier_rmse)      # float: RMSE of inlier correspondences
print(result.converged)        # bool
print(result.iterations)       # int: actual iterations used
```

### `RegistrationMethod` Enum

```python
class RegistrationMethod(IntEnum):
    ICP = 0           # Point-to-point ICP
    ICP_PLANE = 1     # Point-to-plane ICP (requires normals on target)
    NDT = 2           # Normal Distributions Transform
```

### ICP Algorithm

**Point-to-point ICP** (`RegistrationMethod.ICP`):

1. Build KD-tree on target cloud.
2. For each source point, find nearest target point within `max_correspondence_distance`.
3. Compute optimal rigid transform via SVD (Umeyama / Kabsch algorithm).
4. Apply transform to source; repeat until `||T_new - T_old||_F < tolerance` or `max_iterations` reached.

**Point-to-plane ICP** (`RegistrationMethod.ICP_PLANE`):

Same as above, but minimises point-to-plane distance using target normals.  
Requires that `target` has normals computed (RFC-0006).

### NDT Algorithm

Normal Distributions Transform:

1. Partition target cloud into a voxel grid.
2. Compute Gaussian (mean + covariance) for each non-empty voxel.
3. Optimise source→target transform via Newton's method to maximise likelihood.
4. More robust than ICP in partially overlapping or noisy scans.

### KD-Tree Dependency

Use the `kiddo` crate (pure Rust, no unsafe, MIT license) for KD-tree lookups.  
This is the only new Rust dependency introduced by this RFC.

```toml
[dependencies]
kiddo = "^4.2"
```

### `RegistrationResult` Python Class

```python
@dataclass
class RegistrationResult:
    transformation: np.ndarray  # (4, 4) float32
    fitness: float
    inlier_rmse: float
    converged: bool
    iterations: int
```

### Transform Application

After registration, apply the result directly:

```python
source_aligned = source.transform(result.transformation)
```

This reuses the existing `CoordinateTransform` trait (RFC-0001).

---

## Implementation Plan

- [ ] Add `kiddo` dependency to `Cargo.toml`
- [ ] Implement `src/registration/mod.rs` with `RegistrationMethod` enum and `RegistrationResult` struct
- [ ] Implement `src/registration/icp.rs` (point-to-point and point-to-plane)
- [ ] Implement `src/registration/ndt.rs`
- [ ] Expose `PointCloud::register` in `src/lib.rs`
- [ ] Expose `RegistrationResult` and `RegistrationMethod` as PyO3 classes/enums
- [ ] Add `tests/test_registration.py`
- [ ] Benchmark ICP on 100K–1M point pairs; add to `tests/test_benchmark.py`
- [ ] Update `docs/api/` with new `registration.md`

---

## Testing Strategy

```python
# tests/test_registration.py
import numpy as np
import pytest
from pcl_rustic import PointCloud, RegistrationMethod

def make_transformed_pair(n=1000, angle=0.2, translation=(0.5, 0.3, 0.1)):
    """Create source and target that differ by a known transform."""
    import math
    xyz = np.random.randn(n, 3).astype(np.float32)
    target = PointCloud.from_xyz(xyz)

    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    t = np.array(translation, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    source = target.transform(T)
    return source, target, T

def test_icp_convergence():
    source, target, T_gt = make_transformed_pair(n=2000, angle=0.05)
    result = source.register(target, method=RegistrationMethod.ICP,
                             max_correspondence_distance=0.5)
    assert result.converged
    assert result.inlier_rmse < 0.1

def test_icp_recovers_transform():
    source, target, T_gt = make_transformed_pair(n=2000, angle=0.1)
    result = source.register(target, method=RegistrationMethod.ICP,
                             initial_transform=np.eye(4, dtype=np.float32))
    T_est = result.transformation
    # Check translation error
    t_err = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
    assert t_err < 0.05

def test_ndt_convergence():
    source, target, _ = make_transformed_pair(n=5000, angle=0.1)
    result = source.register(target, method=RegistrationMethod.NDT,
                             max_correspondence_distance=1.0)
    assert result.converged

def test_invalid_method_raises():
    source, target, _ = make_transformed_pair()
    with pytest.raises(ValueError):
        source.register(target, method=99)

@pytest.mark.slow
def test_icp_1m_points():
    source, target, _ = make_transformed_pair(n=1_000_000, angle=0.05)
    result = source.register(target, method=RegistrationMethod.ICP)
    assert result.converged
```

---

## Alternatives Considered

- **Expose Open3D registration via Python wrapper**: heavy dependency (~500 MB), breaks zero-copy goal.
- **Pure Python ICP**: too slow for production workloads.
- **Use `pcl` C++ library via FFI**: extreme build complexity.

---

## Drawbacks

- KD-tree construction is O(N log N); for very large point clouds this may be a bottleneck.
- NDT requires careful voxel size tuning; bad voxel sizes cause divergence.
- Point-to-plane ICP requires pre-computed normals (RFC-0006); if not present, it silently falls back to point-to-point.

---

## Unresolved Questions

- Should we expose a multi-scale (coarse-to-fine) ICP variant?
- Should `RegistrationResult` expose the correspondence pairs for debug visualisation?
- What convergence criterion is used for NDT (gradient norm vs. transformation change)?

---

## Related RFCs

- [RFC-0001](./RFC-0001-foundation.md) — CoordinateTransform trait is reused for applying the result
- [RFC-0002](./RFC-0002-tensor-backend.md) — GPU-accelerated KNN distance computation
- [RFC-0006](./RFC-0006-normal-estimation.md) — Point-to-plane ICP requires normals from RFC-0006
