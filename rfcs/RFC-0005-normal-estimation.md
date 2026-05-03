# RFC-0005 — Normal Vector Estimation

| Field | Value |
|-------|-------|
| RFC | 0005 |
| Status | Open for Review |
| Author | liuzhen19 |
| Created | 2026-05-03 |
| Updated | 2026-05-03 |

## Summary

Add normal vector estimation to PCL Rustic. Normals are surface tangent-plane perpendiculars computed per point from the local neighbourhood. They are a prerequisite for RFC-0003 (Normal-based downsampling), RFC-0004 (ICP point-to-plane), and RFC-0006 (segmentation).

## Motivation

Normal vectors encode local surface geometry and are required by:

- **Meshing** (Poisson surface reconstruction, ball-pivoting)
- **Registration** (point-to-plane ICP — see RFC-0004)
- **Segmentation** (region growing — see RFC-0006)
- **Feature description** (FPFH, SHOT descriptors)
- **Downsampling** (normal-based strategy — see RFC-0003)

Most Python point cloud pipelines call Open3D's `estimate_normals()` and then copy the result back. Adding native Rust normal estimation eliminates this round-trip.

## Design

### Algorithm

**Principal Component Analysis (PCA) on local neighbourhood**:

1. For each point `p`, find its K nearest neighbours (k-d tree from `kiddo`).
2. Compute the 3×3 covariance matrix of the neighbourhood.
3. The normal is the eigenvector corresponding to the **smallest** eigenvalue.
4. Orient normals consistently (view-point or minimum spanning tree orientation).

This is the standard approach used by PCL, Open3D, and virtually all point cloud toolkits.

Time complexity: O(N × K) for k-d tree queries + O(N × K) for covariance + O(N) for eigenvectors (3×3 — constant time via Cardano's formula or Jacobi iteration).

### New trait

```rust
// src/traits/normal.rs
pub trait NormalEstimation: Sized {
    /// Estimate per-point normal vectors.
    ///
    /// Stores results in the "nx", "ny", "nz" attribute channels and returns
    /// a new point cloud with those attributes set.
    fn estimate_normals(
        &self,
        config: &NormalEstimationConfig,
    ) -> Result<Self, PointCloudError>;

    /// Returns true if the point cloud has "nx", "ny", "nz" attributes.
    fn has_normals(&self) -> bool;

    /// Returns the normal vectors as an (N, 3) f32 numpy array.
    fn get_normals(&self) -> Result<Vec<Vec<f32>>, PointCloudError>;
}

pub struct NormalEstimationConfig {
    pub k_neighbours: usize,          // default: 30
    pub max_radius:   Option<f32>,    // optional radius bound
    pub view_point:   Option<[f32; 3]>, // for consistent orientation
}
```

### Storage

Normals are stored as three named attribute channels `"nx"`, `"ny"`, `"nz"` in the existing `HashMap<String, Vec<f32>>`. This avoids adding new fields to `HighPerformancePointCloud` and keeps backward compatibility.

```rust
// After estimation:
self.attributes.insert("nx".to_owned(), normals_x);
self.attributes.insert("ny".to_owned(), normals_y);
self.attributes.insert("nz".to_owned(), normals_z);
```

### Python API

```python
import numpy as np
from pcl_rustic import PointCloud, NormalEstimationConfig

# Estimate normals with default config (K=30)
pc_with_normals = pc.estimate_normals()

# Custom config
config = NormalEstimationConfig(k_neighbours=20, view_point=[0.0, 0.0, 1.0])
pc_with_normals = pc.estimate_normals(config=config)

# Check and retrieve
print(pc_with_normals.has_normals())     # True
normals = pc_with_normals.get_normals()  # numpy array (N, 3), dtype float32

# Normals are also accessible as named attributes
nx = pc_with_normals.get_attribute("nx")  # numpy array (N,)
```

### 3×3 eigenvector computation

Use Cardano's formula (closed-form for symmetric 3×3 matrices) to avoid a full eigensolver dependency. This is O(1) per point and branch-free, making it SIMD-friendly.

```rust
// src/utils/eigen3.rs
/// Compute eigenvalues and eigenvectors of a real symmetric 3×3 matrix.
/// Returns (eigenvalues sorted ascending, column eigenvectors).
pub fn sym3_eigen(m: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]);
```

### Normal orientation

Two strategies:

1. **View-point** — flip each normal so it points toward a given view-point (e.g., sensor position `[0, 0, 0]`). Fast and simple.
2. **Minimum spanning tree (MST)** — propagate orientation from a seed point using a Euclidean MST. Consistent but expensive. Deferred to a follow-up RFC.

Default: view-point at `[0.0, 0.0, 0.0]` if no view-point is specified.

### Justfile additions

```just
# Run normal estimation tests
test-normals:
    uv run pytest tests/test_normals.py -v
```

### Integration tests

```python
# tests/test_normals.py
import numpy as np
import pytest
from pcl_rustic import PointCloud, NormalEstimationConfig

@pytest.fixture
def flat_plane():
    """A flat XY plane — all normals should be [0, 0, ±1]."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(-1, 1, (500, 2)).astype(np.float32)
    z = np.zeros((500, 1), dtype=np.float32)
    xyz = np.concatenate([xy, z], axis=1)
    return PointCloud.from_xyz(xyz)

def test_has_normals_false_initially(flat_plane):
    assert not flat_plane.has_normals()

def test_estimate_normals_sets_flag(flat_plane):
    pc = flat_plane.estimate_normals()
    assert pc.has_normals()

def test_normals_shape(flat_plane):
    pc = flat_plane.estimate_normals()
    normals = pc.get_normals()
    assert normals.shape == (500, 3)

def test_flat_plane_normals_are_vertical(flat_plane):
    pc = flat_plane.estimate_normals(
        NormalEstimationConfig(k_neighbours=15, view_point=[0.0, 0.0, 1.0])
    )
    normals = pc.get_normals()
    # Z component should dominate
    assert np.mean(np.abs(normals[:, 2])) > 0.9

def test_normals_unit_length(flat_plane):
    pc = flat_plane.estimate_normals()
    normals = pc.get_normals()
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)
```

## Drawbacks

- k-d tree build time is O(N log N); for real-time applications a radius-search grid may be preferable.
- Storing normals as three separate attribute channels (`nx`, `ny`, `nz`) is correct but slightly verbose; a future RFC could introduce a structured `Vec<[f32; 3]>` attribute type.
- MST-based normal orientation is deferred; view-point orientation may produce artefacts for closed surfaces.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| Integral images (structured point clouds) | Only works for organised (depth image) point clouds |
| Moving Least Squares (MLS) | More accurate but O(N × K²); better suited to surface smoothing |
| GPU PCA via Burn tensors | Possible future optimisation; requires RFC-0002 |

## Unresolved Questions

1. Should `estimate_normals` mutate in place or return a new `PointCloud`? (RFC-0001 convention: return new.)
2. What happens if a point has fewer than 3 neighbours? (Degenerate case — set normal to `[0, 0, 1]` and warn?)
3. Should we provide a `smooth_normals` post-processing step?
4. Should Cardano's formula be replaced by LAPACK bindings for better numerical stability on near-degenerate matrices?

## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — attribute channel storage (`HashMap`) used for normal data
- [RFC-0003 — Advanced Downsampling](./RFC-0003-advanced-downsampling.md) — `NORMAL_BASED` strategy consumes normals from this RFC
- [RFC-0004 — Point Cloud Registration](./RFC-0004-point-cloud-registration.md) — point-to-plane ICP requires target normals from this RFC
- [RFC-0006 — Point Cloud Segmentation](./RFC-0006-point-cloud-segmentation.md) — region-growing segmentation uses normals
- [RFC-0007 — Testing & CI Infrastructure](./RFC-0007-testing-ci-infrastructure.md) — integration test conventions

## Reviews

### Review 1 — Sub-agent Alpha

> **Status**: APPROVED
>
> The PCA-based approach is correct and the choice to store normals in the existing attribute map is elegant — it avoids schema changes while remaining interoperable with `get_attribute("nx")`. The Cardano's formula approach for 3×3 eigenvectors is the right call for performance. Strongly endorse making the view-point configurable (default `[0,0,0]`). The integration test for flat-plane normal estimation is excellent. One suggestion: add a test for a sphere where normals should be radially outward, to catch orientation bugs.

### Review 2 — Sub-agent Beta

> **Status**: APPROVED (with minor comments)
>
> The storage design using `"nx"`, `"ny"`, `"nz"` attributes is clean. Unresolved Question 1 (mutate vs return new) should be answered: the existing convention (confirmed in RFC-0001) is to return a new `PointCloud`. Stick to that for consistency. Unresolved Question 2 (fewer than 3 neighbours) must be resolved before implementation; panicking is not acceptable. Recommend returning a zero normal `[0,0,0]` with a logged warning, and documenting this behaviour.
