# RFC-0003: Additional Downsampling Strategies

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Draft                              |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2026-05-03                         |
| Updated | 2026-05-03                         |

## Summary

Extend `DownsampleStrategy` beyond the current `RANDOM` and `CENTROID` options to include **Farthest Point Sampling (FPS)** and **Normal-based downsampling**. Both strategies produce geometrically representative subsets critical for registration, reconstruction, and learning tasks.

## Motivation

The current voxel-grid downsampling with `RANDOM` or `CENTROID` strategies is fast but produces uniform spatial coverage only when the underlying point distribution is roughly uniform. Real-world LiDAR data contains dense near-field regions and sparse far-field regions; for downstream tasks like ICP registration or surface reconstruction:

- **FPS** ensures maximum geometric coverage regardless of density distribution.
- **Normal-based** preserves feature-rich regions (edges, corners) by sampling proportionally to local curvature/normal variation.

Both strategies are widely used in point-cloud deep learning (PointNet++, Point Transformer) and classical 3D processing pipelines.

## Design Details

### Extended DownsampleStrategy Enum

```rust
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum DownsampleStrategy {
    Random,
    Centroid,
    FarthestPoint,       // NEW
    NormalBased,         // NEW — requires normals to be pre-computed
}
```

Python usage:

```python
from pcl_rustic import DownsampleStrategy

# Farthest Point Sampling — returns exactly `n_points` points
pc_fps = pc.farthest_point_sample(n_points=4096)

# Normal-based — voxel grid weighted by local normal variation
pc_normal = pc.voxel_downsample(
    voxel_size=0.1,
    strategy=DownsampleStrategy.NormalBased,
)
```

### Farthest Point Sampling (FPS)

**Algorithm**:
1. Select a random seed point.
2. Iteratively select the point with the maximum minimum distance to the already-selected set.
3. Repeat until `n_points` are selected.

**Complexity**: O(N × k) where k = `n_points`. For large N and k this is expensive; GPU acceleration (see [RFC-0002](./RFC-0002-gpu-acceleration.md)) is critical.

**New Python method**:

```python
def farthest_point_sample(self, n_points: int, seed: int | None = None) -> PointCloud:
    """
    Downsample using Farthest Point Sampling.

    Args:
        n_points: Number of points to select.
        seed: Optional random seed for reproducibility.

    Returns:
        A new PointCloud with exactly `n_points` points.

    Raises:
        ValueError: If `n_points` > current point count.
    """
```

**Implementation notes**:
- CPU: implemented in Rust with `rayon` parallelism over distance computation.
- GPU: Burn tensor `argmax` / `gather` operations (enabled by [RFC-0002](./RFC-0002-gpu-acceleration.md)).

### Normal-based Downsampling

**Algorithm**: Within each voxel, select the point whose local normal deviates most from the voxel's average normal. This preserves edge and corner points over flat-surface interior points.

**Prerequisite**: Normals must be pre-computed (see [RFC-0005](./RFC-0005-normal-estimation.md)). If the point cloud has no normals, fall back to `CENTROID` with a warning.

**New Python method** (extends existing `voxel_downsample`):

```python
pc.voxel_downsample(
    voxel_size=0.1,
    strategy=DownsampleStrategy.NormalBased,
)
```

### Voxel Downsampling API Update

The existing `voxel_downsample` signature remains unchanged for `RANDOM` / `CENTROID`. The `NormalBased` strategy inspects the point cloud's `normals` attribute.

### Performance Targets

| Strategy      | 10 M pts | 50 M pts | GPU speedup (est.) |
|---------------|----------|----------|--------------------|
| RANDOM        | <1 s     | <5 s     | ~2×                |
| CENTROID      | 7.7 s    | 47 s     | ~5×                |
| FPS (k=4096)  | ~5 s     | ~25 s    | ~10×               |
| NormalBased   | ~9 s     | ~50 s    | ~5×                |

## Alternatives Considered

### Alternative 1: Poisson Disk Sampling

Ensures minimum inter-point distance. Rejected for this RFC because:
- More complex to implement efficiently at scale.
- Spatial data structures (octree) not yet part of the architecture.
- Deferred to a future RFC.

### Alternative 2: Grid Sampling with Density Weighting

Weight voxel selection probability by local point density. Rejected as it lacks the strong geometric coverage guarantees of FPS.

## Open Questions

1. **FPS on very large clouds (>50 M)**: Is a hierarchical / coarse-to-fine FPS acceptable for performance, or must we guarantee exact FPS?
2. **Seed reproducibility across backends**: FPS with `seed` must produce identical results on CPU and GPU backends. This requires careful RNG handling in Burn.
3. **NormalBased without normals**: Should the library auto-compute normals with a default `k=20`, or hard-fail with an error?

## Implementation Plan

- [ ] Add `FarthestPoint` and `NormalBased` variants to `DownsampleStrategy` enum
- [ ] Implement `farthest_point_sample(n_points, seed)` in `src/point_cloud/fps.rs` (new file)
- [ ] Implement normal-based voxel selection in `src/point_cloud/voxel.rs`
- [ ] Expose `farthest_point_sample` Python method via PyO3
- [ ] Add pytest tests: `tests/test_downsample.py`
  - `test_fps_exact_count()` — output has exactly `n_points` points
  - `test_fps_coverage()` — verify max pairwise distance is larger than random sample
  - `test_fps_seed_reproducibility()` — same seed produces same result
  - `test_normal_based_no_normals_warning()` — warns and falls back to centroid
  - `test_normal_based_preserves_edges()` — edge points appear more often than flat-surface points
- [ ] Update `justfile`: no new recipe needed; covered by `just test`
- [ ] Update `rfcs/README.md` index status

## Related RFCs

- **Background**: [RFC-0001](./RFC-0001-initial-architecture.md) — existing downsampling design
- **Accelerated by** ← [RFC-0002](./RFC-0002-gpu-acceleration.md): GPU backend for FPS inner loop
- **Depends on** → [RFC-0005](./RFC-0005-normal-estimation.md): normals required for NormalBased strategy
- **Relevant for** → [RFC-0004](./RFC-0004-point-cloud-registration.md): FPS used for registration keypoint selection

## References

- Qi et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (NeurIPS 2017)
- [Open3D FPS documentation](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)
- [RFC-0001](./RFC-0001-initial-architecture.md)
- [RFC-0002](./RFC-0002-gpu-acceleration.md)
- [RFC-0005](./RFC-0005-normal-estimation.md)
