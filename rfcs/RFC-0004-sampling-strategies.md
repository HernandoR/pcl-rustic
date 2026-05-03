# RFC-0004: Advanced Downsampling Strategies

- **Status**: Draft
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-05-03
- **Updated**: 2026-05-03

---

## Summary

Add Farthest Point Sampling (FPS) and Normal-weighted downsampling to pcl-rustic's voxel-grid framework, and introduce a separate non-voxel spatial sampling API.  
These strategies are standard inputs for deep learning point cloud models (PointNet++, SparseConvNet).

---

## Motivation

The existing voxel downsampling only offers:
- `RANDOM` — random representative per voxel
- `CENTROID` — point nearest to voxel centroid

Production needs identified in issues and the README roadmap:
1. **FPS** — used by PointNet++, ensures uniform spatial coverage; cannot be expressed as a per-voxel strategy.
2. **Normal-weighted** — prioritise points at geometric discontinuities; improves surface reconstruction quality.
3. **Poisson Disk** — ensures minimum inter-point distance; useful for mesh remeshing inputs.

---

## Detailed Design

### Revised Strategy Taxonomy

```
DownsampleStrategy (existing, voxel-relative)
  RANDOM
  CENTROID
  NORMAL_WEIGHTED   ← new

SpatialSampleMethod (new, global)
  FPS               ← Farthest Point Sampling
  POISSON_DISK      ← Poisson Disk Sampling
```

### Voxel-level Normal-Weighted Strategy

Within each voxel, select the point with the highest **local normal variation** (proxy for surface curvature):

```rust
pub struct NormalWeightedStrategy;

impl VoxelSampleStrategy for NormalWeightedStrategy {
    fn select_representative(
        &self,
        indices: &[usize],
        xyz: &[[f32; 3]],
        normals: Option<&[[f32; 3]]>,
    ) -> Result<usize> {
        // If normals available: pick point with largest angular deviation from mean normal
        // Fallback to centroid if no normals
    }
}
```

### FPS (Global Spatial Sampling)

FPS does not decompose into independent per-voxel choices; it requires a global distance matrix.  
New method on `PointCloud`:

```python
pc_fps = pc.farthest_point_sample(n_samples: int) -> PointCloud
```

Rust implementation:
1. Pick a random seed point.
2. Maintain a `distances: Vec<f32>` array (minimum distance of each point to the selected set).
3. Iteratively select the point with the maximum distance; update `distances`.
4. Time: O(N·K) where K = `n_samples`.
5. GPU acceleration path (RFC-0002): parallelise the distance update step.

### Poisson Disk Sampling

```python
pc_pd = pc.poisson_disk_sample(min_distance: float, seed: int = 0) -> PointCloud
```

Algorithm: dart-throwing with a spatial hash grid (similar to voxel grid but rejection-based).

### Python API Summary

```python
from pcl_rustic import DownsampleStrategy, PointCloud

# Existing (unchanged)
pc.voxel_downsample(voxel_size=0.1, strategy=DownsampleStrategy.RANDOM)
pc.voxel_downsample(voxel_size=0.1, strategy=DownsampleStrategy.CENTROID)

# New voxel strategy
pc.voxel_downsample(voxel_size=0.1, strategy=DownsampleStrategy.NORMAL_WEIGHTED)

# New global methods
pc.farthest_point_sample(n_samples=2048)
pc.poisson_disk_sample(min_distance=0.05, seed=42)
```

### `DownsampleStrategy` Enum Extension

```python
class DownsampleStrategy(IntEnum):
    RANDOM = 0          # existing
    CENTROID = 1        # existing
    NORMAL_WEIGHTED = 2 # new
```

---

## Implementation Plan

- [ ] Add `NormalWeightedStrategy` struct in `src/point_cloud/voxel.rs`
- [ ] Add `NORMAL_WEIGHTED = 2` to the PyO3 `DownsampleStrategy` enum
- [ ] Implement `farthest_point_sample` in `src/point_cloud/fps.rs`
- [ ] Implement `poisson_disk_sample` in `src/point_cloud/poisson.rs`
- [ ] Expose new methods in `src/lib.rs` PyO3 bindings
- [ ] Add pytest tests: `tests/test_sampling.py`
- [ ] Benchmark FPS on 1M and 10M point clouds; add to `tests/test_benchmark.py`
- [ ] Update `docs/api/downsample.md`

---

## Testing Strategy

```python
# tests/test_sampling.py
import pytest, numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

@pytest.fixture
def grid_cloud():
    """Regular 3D grid cloud for deterministic tests."""
    coords = np.mgrid[0:10, 0:10, 0:10].reshape(3, -1).T.astype(np.float32)
    return PointCloud.from_xyz(coords)

def test_normal_weighted_strategy(grid_cloud):
    result = grid_cloud.voxel_downsample(2.0, DownsampleStrategy.NORMAL_WEIGHTED)
    assert 0 < result.point_count() <= grid_cloud.point_count()

def test_fps_correct_count(grid_cloud):
    n = 50
    result = grid_cloud.farthest_point_sample(n)
    assert result.point_count() == n

def test_fps_spatial_coverage(grid_cloud):
    result = grid_cloud.farthest_point_sample(100)
    xyz = result.get_xyz()
    # Verify no two sampled points are too close
    from scipy.spatial.distance import pdist
    dists = pdist(xyz)
    # FPS guarantees maximin spacing — minimum distance should be non-trivial
    assert dists.min() > 0.5

def test_poisson_disk_min_distance():
    xyz = np.random.randn(5000, 3).astype(np.float32) * 10
    pc = PointCloud.from_xyz(xyz)
    min_d = 0.5
    result = pc.poisson_disk_sample(min_distance=min_d)
    from scipy.spatial.distance import pdist
    dists = pdist(result.get_xyz())
    assert dists.min() >= min_d - 1e-4

@pytest.mark.slow
def test_fps_10m_points():
    xyz = np.random.randn(10_000_000, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    result = pc.farthest_point_sample(4096)
    assert result.point_count() == 4096
```

---

## Alternatives Considered

- **Expose FPS only as a Python loop calling `voxel_downsample` repeatedly**: too slow; O(N²) per iteration.
- **Use Open3D's FPS via Python interop**: adds a large optional dependency; defeats the zero-dependency goal.
- **Grid-based FPS approximation**: faster but loses the exact maximin guarantee needed for PointNet++ inputs.

---

## Drawbacks

- FPS is O(N·K) and not parallelisable without the GPU backend (RFC-0002); on CPU it may be slow for very large N.
- `NORMAL_WEIGHTED` requires pre-computed normals (RFC-0006) to be most effective; without normals it degrades to centroid.

---

## Unresolved Questions

- Should FPS preserve all attributes (intensity, RGB, custom) in the output, or only XYZ?
- Should we expose a `farthest_point_sample_indices` method that returns `np.ndarray[int]` indices into the original cloud?
- What is the seed behaviour for reproducibility in `poisson_disk_sample`?

---

## Related RFCs

- [RFC-0001](./RFC-0001-foundation.md) — VoxelDownsample trait and DownsampleStrategy enum are defined there
- [RFC-0002](./RFC-0002-tensor-backend.md) — FPS distance updates can be GPU-accelerated
- [RFC-0006](./RFC-0006-normal-estimation.md) — NORMAL_WEIGHTED strategy uses normals computed by RFC-0006
