# RFC-0006: Normal Vector Estimation

- **Status**: Draft
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-05-03
- **Updated**: 2026-05-03

---

## Summary

Add surface normal estimation to pcl-rustic via PCA on K-nearest neighbours or radius search.  
Normals are stored as a first-class attribute (`nx`, `ny`, `nz`) and exposed through a consistent Python API.  
They are a prerequisite for Point-to-Plane ICP (RFC-0005), Normal-Weighted downsampling (RFC-0004), and segmentation (RFC-0007).

---

## Motivation

Normals are the most commonly needed derived attribute in 3D processing:
- **Registration**: Point-to-plane ICP reduces to ~0.1× the iterations of point-to-point.
- **Sampling**: Normal-weighted voxel sampling preserves edge points better.
- **Segmentation**: RANSAC plane/cylinder fitting, region growing, and deep learning models all use normals.
- **Rendering**: Shading requires surface normals.

Computing normals in Python (numpy KNN + PCA) for 10M points takes ~60 s.  
A Rust implementation with `kiddo` KD-tree and rayon parallel PCA should be ~1–2 s.

---

## Detailed Design

### Normal Estimation Method

For each point `p_i`, collect its K nearest neighbours (or all within radius `r`), compute the 3×3 covariance matrix, and take the eigenvector corresponding to the smallest eigenvalue as the normal direction.

```python
pc.estimate_normals(
    k: int = 30,                       # KNN search radius
    *,
    radius: float | None = None,       # alternative: radius search (overrides k)
    orient_towards: np.ndarray | None = None,  # (3,) viewpoint for consistent orientation
) -> None  # stores normals in-place as "nx", "ny", "nz" attributes
```

### Normal Storage

Normals are stored as three `float32` attribute columns: `nx`, `ny`, `nz`.

```python
pc.has_normals() -> bool
pc.get_normals() -> np.ndarray  # shape (N, 3), dtype float32
pc.set_normals(normals: np.ndarray)  # shape (N, 3), dtype float32
```

These are convenience wrappers around the existing `add_attribute`/`get_attribute` API.

### Orientation Consistency

Without a viewpoint, PCA normals are ambiguous (may point inward or outward).  
When `orient_towards` (a 3D viewpoint, e.g., LiDAR origin) is provided:

```
if dot(normal_i, viewpoint - p_i) < 0:
    normal_i = -normal_i
```

For organised point clouds a propagation-based orientation using a minimum spanning tree (Hoppe 1992) will be a future enhancement.

### Rust Implementation

```rust
// src/normals/mod.rs
pub fn estimate_normals(
    xyz: &[[f32; 3]],
    k: usize,
    viewpoint: Option<[f32; 3]>,
) -> Vec<[f32; 3]> {
    // 1. Build KD-tree (kiddo)
    // 2. For each point, query K neighbours
    // 3. Compute PCA of neighbour positions (subtract mean, 3×3 covariance)
    // 4. Eigendecompose: smallest eigenvalue's eigenvector = normal
    // 5. Orient if viewpoint given
    // 6. Return Vec<[f32; 3]> normals
}
```

Parallelise step 2–5 with `rayon::par_iter()`.

### Eigendecomposition

Use `nalgebra` (already transitive through burn) or a minimal 3×3 symmetric eigensolver (closed-form Cardano formula) to avoid a heavy linear algebra dependency.

---

## Implementation Plan

- [ ] Add `src/normals/mod.rs` with `estimate_normals` function
- [ ] Add `has_normals`, `get_normals`, `set_normals` to `HighPerformancePointCloud`
- [ ] Expose `estimate_normals` in `src/lib.rs` PyO3 bindings
- [ ] Integrate normals into `NormalWeightedStrategy` (RFC-0004)
- [ ] Wire normals check into ICP_PLANE (RFC-0005)
- [ ] Add `tests/test_normals.py`
- [ ] Add benchmark for normal estimation (1M, 10M) to `tests/test_benchmark.py`
- [ ] Update `docs/api/` with new `normals.md`

---

## Testing Strategy

```python
# tests/test_normals.py
import numpy as np
import pytest
from pcl_rustic import PointCloud

def test_normals_plane():
    """Normals on a flat XY plane should point in Z direction."""
    xy = np.random.uniform(0, 10, (500, 2)).astype(np.float32)
    z = np.zeros((500, 1), dtype=np.float32)
    xyz = np.hstack([xy, z])
    pc = PointCloud.from_xyz(xyz)
    pc.estimate_normals(k=10, orient_towards=np.array([0.0, 0.0, 10.0], dtype=np.float32))
    normals = pc.get_normals()
    assert normals.shape == (500, 3)
    # All normals should be approximately (0, 0, 1) or (0, 0, -1)
    assert np.abs(normals[:, 2]).mean() > 0.95

def test_normals_sphere():
    """Normals on a sphere should be approximately radial."""
    theta = np.random.uniform(0, 2*np.pi, 1000).astype(np.float32)
    phi = np.random.uniform(0, np.pi, 1000).astype(np.float32)
    r = 5.0
    xyz = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi),
    ]).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc.estimate_normals(k=20, orient_towards=np.zeros(3, dtype=np.float32))
    normals = pc.get_normals()
    # Each normal should be approximately the normalised position vector
    pos_normalised = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    dot_products = np.abs((normals * pos_normalised).sum(axis=1))
    assert dot_products.mean() > 0.9

def test_has_normals_false_initially():
    xyz = np.random.randn(10, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    assert not pc.has_normals()

def test_set_normals_manual():
    xyz = np.random.randn(10, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    normals = np.zeros((10, 3), dtype=np.float32)
    normals[:, 2] = 1.0
    pc.set_normals(normals)
    assert pc.has_normals()
    np.testing.assert_allclose(pc.get_normals(), normals)

def test_normals_preserved_after_downsample():
    xyz = np.random.randn(1000, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc.estimate_normals(k=20)
    from pcl_rustic import DownsampleStrategy
    ds = pc.voxel_downsample(0.5, DownsampleStrategy.CENTROID)
    assert ds.has_normals()

@pytest.mark.slow
def test_normals_10m():
    xyz = np.random.randn(10_000_000, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc.estimate_normals(k=30)
    assert pc.has_normals()
    assert pc.get_normals().shape == (10_000_000, 3)
```

---

## Alternatives Considered

- **Python NumPy PCA loop**: ~60 s for 10M points; not acceptable.
- **scipy KD-tree from Python**: avoids new Rust dependency but cannot parallelise with `rayon`.
- **Open3D normals via Python interop**: adds heavyweight dependency.
- **Store normals as separate `PointCloud` object**: confusing API; normals are tightly coupled to the point positions.

---

## Drawbacks

- KD-tree construction takes O(N log N) time; for N > 50M this becomes noticeable.
- Radius-search normals can be empty for isolated points (fallback: assign `[0,0,0]` normal with a warning).
- PCA normals near sharp edges are ill-defined (flat window straddles both surfaces).

---

## Unresolved Questions

- Should normal estimation radius search mode use a spatial hash grid instead of KD-tree for better cache locality?
- Should `set_normals` validate that normals are unit vectors, or silently normalise?
- Should normals survive `to_las` / `from_las` round-trips (LAS 1.4 point formats 6–10 have no normal field)?

---

## Related RFCs

- [RFC-0001](./RFC-0001-foundation.md) — PointCloudProperties trait used for normal attribute storage
- [RFC-0002](./RFC-0002-tensor-backend.md) — GPU-accelerated KNN for faster normal estimation
- [RFC-0004](./RFC-0004-sampling-strategies.md) — NORMAL_WEIGHTED strategy uses normals from this RFC
- [RFC-0005](./RFC-0005-registration.md) — ICP_PLANE mode requires normals from this RFC
- [RFC-0007](./RFC-0007-segmentation.md) — Region-growing segmentation uses normals from this RFC
