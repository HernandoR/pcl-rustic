# RFC-0004 — Point Cloud Registration Algorithms (ICP & NDT)

| Field | Value |
|-------|-------|
| RFC | 0004 |
| Status | Open for Review |
| Author | liuzhen19 |
| Created | 2026-05-03 |
| Updated | 2026-05-03 |

## Summary

Add point cloud registration algorithms to PCL Rustic, starting with **Iterative Closest Point (ICP)** and **Normal Distributions Transform (NDT)**. Both algorithms estimate the rigid-body transformation that best aligns a *source* point cloud to a *target* point cloud.

## Motivation

Point cloud registration is a fundamental operation in:

- 3D mapping and SLAM (simultaneous localisation and mapping)
- Robot perception (aligning LiDAR scans)
- CAD inspection (measuring deviation between as-built and as-designed)

Without native registration support, users must call external libraries (Open3D, PCL C++) from Python, introducing heavy dependencies and copying overhead. Adding ICP and NDT directly in Rust gives users a zero-dependency, high-performance alternative that integrates seamlessly with the existing `PointCloud` API.

## Design

### New trait

```rust
// src/traits/registration.rs
pub trait PointCloudRegistration: Sized {
    /// Align `source` (self) to `target` and return the estimated
    /// rigid transformation and a convergence report.
    fn register(
        &self,
        target: &Self,
        config: &RegistrationConfig,
    ) -> Result<RegistrationResult, PointCloudError>;
}

pub struct RegistrationConfig {
    pub algorithm:       RegistrationAlgorithm,
    pub max_iterations:  usize,        // default: 50
    pub convergence_eps: f32,          // default: 1e-6
    pub max_correspondence_dist: f32,  // ICP: max point-pair distance
    pub voxel_size:      Option<f32>,  // NDT: voxel resolution
    pub initial_transform: Option<[[f32; 4]; 4]>, // optional warm-start
}

pub enum RegistrationAlgorithm {
    PointToPoint,  // classic ICP
    PointToPlane,  // ICP variant using normals (requires RFC-0005)
    Ndt,           // Normal Distributions Transform
}

pub struct RegistrationResult {
    pub transform:    [[f32; 4]; 4],  // 4×4 homogeneous matrix
    pub fitness:      f32,            // mean squared error of correspondences
    pub converged:    bool,
    pub iterations:   usize,
}
```

### ICP — Point-to-Point

Algorithm overview:

1. For each point in *source*, find the nearest neighbour in *target* (k-d tree or brute force).
2. Compute the rigid transform that minimises the sum of squared distances (SVD-based closed-form solution).
3. Apply the transform to *source* and repeat until convergence.

k-d tree: use the `kiddo` crate (pure Rust, no unsafe) for O(N log N) build + O(log N) query.

```toml
# Cargo.toml additions
kiddo = "^4.2"
```

### ICP — Point-to-Plane

Same as point-to-point but the error metric projects onto the target normal at each correspondence. Requires normals on the *target* cloud (see RFC-0005). Converges faster and more accurately on smooth surfaces.

### NDT

1. Voxelise the *target* cloud.
2. For each voxel, compute the Gaussian distribution (mean μ, covariance Σ) of its points.
3. For each point in *source*, evaluate the likelihood against the target voxel's distribution.
4. Optimise the transform using Newton's method to maximise the total likelihood.

NDT is more robust to noise and does not require point-level correspondences.

### Python API

```python
from pcl_rustic import PointCloud, RegistrationConfig, RegistrationAlgorithm

# Default ICP (point-to-point)
result = source.register(target)
print(result.transform)    # 4×4 numpy array (float32)
print(result.converged)    # bool
print(result.fitness)      # mean squared error
print(result.iterations)   # int

# NDT with custom config
config = RegistrationConfig(
    algorithm=RegistrationAlgorithm.NDT,
    max_iterations=100,
    convergence_eps=1e-8,
    voxel_size=0.5,
)
result = source.register(target, config=config)

# Apply the result
aligned = source.transform(result.transform)
```

### Data classes

```python
# Python dataclasses exposed via PyO3
class RegistrationResult:
    transform: np.ndarray   # shape (4, 4), dtype float32
    fitness: float
    converged: bool
    iterations: int

class RegistrationConfig:
    algorithm: RegistrationAlgorithm = RegistrationAlgorithm.POINT_TO_POINT
    max_iterations: int = 50
    convergence_eps: float = 1e-6
    max_correspondence_dist: float = float("inf")
    voxel_size: float | None = None
    initial_transform: np.ndarray | None = None  # shape (4, 4)
```

### Justfile additions

```just
# Run registration integration tests
test-registration:
    uv run pytest tests/test_registration.py -v

# Benchmark ICP vs NDT
benchmark-registration:
    uv run pytest tests/test_benchmark.py -k registration -v -s
```

### Integration tests

```python
# tests/test_registration.py
import numpy as np
import pytest
from pcl_rustic import PointCloud, RegistrationConfig, RegistrationAlgorithm

@pytest.fixture
def source_target_pair():
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((2000, 3)).astype(np.float32)
    source = PointCloud.from_xyz(xyz)

    # Known transform: rotate 5° around Z, translate (0.1, 0.2, 0.0)
    theta = np.radians(5)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ], dtype=np.float32)
    t = np.array([0.1, 0.2, 0.0], dtype=np.float32)
    target = source.rigid_transform(R, t)
    return source, target, R, t

def test_icp_converges(source_target_pair):
    source, target, _, _ = source_target_pair
    result = source.register(target)
    assert result.converged
    assert result.fitness < 0.01

def test_icp_recovers_transform(source_target_pair):
    source, target, R_gt, t_gt = source_target_pair
    result = source.register(target)
    T = result.transform
    t_est = T[:3, 3]
    np.testing.assert_allclose(t_est, t_gt, atol=0.05)

@pytest.mark.slow
def test_ndt_converges(source_target_pair):
    source, target, _, _ = source_target_pair
    config = RegistrationConfig(
        algorithm=RegistrationAlgorithm.NDT,
        voxel_size=0.5,
    )
    result = source.register(target, config=config)
    assert result.converged
```

## Drawbacks

- ICP requires a good initial alignment; diverges if the starting transformation is too far from the true solution.
- NDT is significantly more complex to implement correctly (covariance estimation, numerical stability of Σ⁻¹).
- Adding `kiddo` as a dependency increases compile time.
- Point-to-plane ICP depends on RFC-0005 (normals); cannot be fully tested until RFC-0005 is implemented.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| Wrap Open3D's ICP | Heavy C++ dependency; breaks zero-dependency goal |
| GICP / VGICP | More accurate but significantly more complex; defer to future RFC |
| FGR (Fast Global Registration) | Good for initial alignment but no convergence guarantee |

## Unresolved Questions

1. Should `RegistrationConfig` support a `correspondence_estimator` plugin (like `DownsampleStrategy`)?
2. What k-d tree crate is best? `kiddo` vs `kd-tree` vs `nanoflann-sys`?
3. Should convergence be measured on transform delta or fitness delta?
4. How to handle degenerate clouds (< 6 points)?
5. Should NDT expose intermediate iteration callbacks for visualisation?

## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — `CoordinateTransform` trait is reused for applying the estimated transform
- [RFC-0002 — GPU Acceleration](./RFC-0002-gpu-acceleration.md) — iterative algorithms benefit most from GPU; batched correspondence search
- [RFC-0005 — Normal Estimation](./RFC-0005-normal-estimation.md) — point-to-plane ICP requires target normals
- [RFC-0007 — Testing & CI Infrastructure](./RFC-0007-testing-ci-infrastructure.md) — slow-marker strategy for expensive registration tests

## Reviews

### Review 1 — Sub-agent Alpha

> **Status**: APPROVED
>
> The design is solid. Using `kiddo` for k-d tree is a good choice — it is the most actively maintained pure-Rust k-d tree crate. The SVD-based closed-form solution for point-to-point ICP is correct and well-understood. The Python API is clean and follows existing conventions. Unresolved Question 2 (k-d tree crate) should be resolved before implementation begins; recommend `kiddo` v4 based on benchmarks. One suggestion: add a `max_correspondence_dist` clamp to the point-to-point fitness computation to avoid outlier pairs dominating the loss.

### Review 2 — Sub-agent Beta

> **Status**: APPROVED (with minor comments)
>
> NDT design is correct at a high level. The Newton's method optimiser for NDT can be tricky to implement stably — recommend referencing the Biber & Straßer (2003) paper and the `ndt_cpu` implementation in Autoware as a correctness reference. The integration test using a synthetic 5° rotation is excellent for CI (deterministic and fast). Agree with Unresolved Question 4: degenerate cloud handling must be part of the implementation to avoid panics. Also suggest documenting the expected runtime complexity (O(N log N) for ICP, O(N/K + M log M) for NDT where K is voxel density and M is number of voxels).
