# RFC-0003: Advanced Downsampling Strategies

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-05-03 |
| **Updated** | 2026-05-03 |
| **Related RFCs** | [RFC-0001](RFC-0001-core-architecture.md) · [RFC-0005](RFC-0005-normal-estimation.md) · [RFC-0006](RFC-0006-gpu-acceleration.md) · [RFC-0007](RFC-0007-testing-infrastructure.md) |

## Summary

Extend the downsampling subsystem beyond the current `RANDOM` and `CENTROID` voxel strategies to include **Furthest Point Sampling (FPS)**, **Normal-Based Downsampling**, **Poisson-Disk Sampling**, and **Adaptive Voxel Grid** (LOD-aware). All strategies implement the existing `DownsampleStrategy` Trait; the `voxel_downsample` API gains an optional `seed` parameter for reproducibility.

## Motivation

The voxel-grid downsampler in RFC-0001 produces spatially uniform results but has known weaknesses:

1. **Uneven feature preservation**: small voxels oversample flat regions; large voxels destroy fine geometric features.
2. **No curvature/normal awareness**: equal density is not equal information density for surfaces.
3. **Non-reproducible**: `RANDOM` strategy yields different results on each run.
4. **No coverage guarantee**: Poisson-disk sampling provides a minimum separation guarantee needed for meshing.

### Use Cases

- 3D object detection models require FPS for spatially diverse input batches.
- Surface reconstruction algorithms (Poisson, Marching Cubes) require Poisson-disk sampled inputs.
- LOD (Level-of-Detail) streaming requires adaptive voxel grids that maintain feature density.
- Registration and segmentation algorithms benefit from normal-aware sampling.

## Design

### 4.1 New Strategy Enum Values

```python
class DownsampleStrategy:
    RANDOM      = 0   # existing
    CENTROID    = 1   # existing
    FPS         = 2   # new: furthest point sampling
    NORMAL      = 3   # new: normal-based (requires normals; see RFC-0005)
    POISSON     = 4   # new: Poisson-disk minimum separation guarantee
    ADAPTIVE    = 5   # new: adaptive voxel grid (curvature-aware)
```

### 4.2 Furthest Point Sampling (FPS)

FPS selects `k` points that maximise the minimum pairwise distance:

```python
# Python API
pc_fps = pc.fps_downsample(num_points=1024, seed=42)
```

Algorithm (iterative greedy, O(N·k)):
1. Pick a random seed point.
2. Maintain per-point distance-to-selected-set array `D`.
3. Greedily pick the point `argmax(D)`, update `D`.

The Rust implementation uses `rayon` for parallel distance updates.

```rust
pub trait Downsample {
    fn fps_downsample(
        &self,
        num_points: usize,
        seed: Option<u64>,
    ) -> Result<HighPerformancePointCloud>;
}
```

> **Note**: FPS depends on Normal estimation (RFC-0005) when combined with NORMAL strategy.

### 4.3 Normal-Based Downsampling

When normals are available (see [RFC-0005](RFC-0005-normal-estimation.md)), allocate sample budget proportional to local curvature:

```python
# Requires normals to be pre-computed
pc.estimate_normals(k=20)
pc_norm = pc.voxel_downsample(0.05, DownsampleStrategy.NORMAL)
```

Implementation: compute curvature as the ratio of smallest/largest PCA eigenvalue in each voxel; multiply the inverse voxel count by a curvature weight.

### 4.4 Poisson-Disk Sampling

Guarantees that no two output points are closer than `min_dist`:

```python
pc_pd = pc.poisson_disk_downsample(min_dist=0.05, seed=42)
```

Algorithm: shuffle input, greedily accept points if no accepted point is within `min_dist` (spatial hash for O(N) amortised).

### 4.5 Adaptive Voxel Grid

Subdivide voxels recursively until each leaf voxel has ≤ `target_density` points per unit volume:

```python
pc_adp = pc.adaptive_voxel_downsample(target_density=100.0, min_voxel=0.01, max_voxel=1.0)
```

Implemented as an octree; exposed as a method rather than through the `voxel_downsample` strategy parameter because it does not fit the fixed-voxel-size contract.

### 4.6 Reproducibility: `seed` Parameter

Add an optional `seed: Option<u64>` to `voxel_downsample` and all new methods. When `None`, behaviour remains stochastic (backward-compatible).

```python
pc_down = pc.voxel_downsample(0.1, DownsampleStrategy.RANDOM, seed=0)
```

### 4.7 API Changes Summary

```python
# Extended signatures (new params are keyword-only, default None/unchanged)
pc.voxel_downsample(voxel_size: float, strategy: DownsampleStrategy, *, seed: int | None = None) -> PointCloud
pc.fps_downsample(num_points: int, *, seed: int | None = None) -> PointCloud
pc.poisson_disk_downsample(min_dist: float, *, seed: int | None = None) -> PointCloud
pc.adaptive_voxel_downsample(target_density: float, *, min_voxel: float = 0.01, max_voxel: float = 1.0) -> PointCloud
```

### Alternatives Considered

| Approach | Reason Not Chosen |
|----------|-------------------|
| Port Open3D's C++ FPS directly | Licence issues; C++ dependency breaks pure-Rust build |
| GPU-only FPS | Adds GPU requirement; CPU path needed for portability |
| Expose octree as a standalone class | Adds API surface; internal use suffices for now |

## Impact

### Breaking Changes

- `voxel_downsample(voxel_size, strategy)` gains an optional keyword `seed`; fully backward-compatible.
- New `DownsampleStrategy` int values (2–5) are additive.

### Performance Implications

| Algorithm | Complexity | Expected throughput (1M pts) |
|-----------|-----------|------------------------------|
| FPS | O(N·k) | ~0.5 M pts/s (CPU) |
| Normal | O(N) + normal cost | ~1 M pts/s |
| Poisson-disk | O(N) amortised | ~2 M pts/s |
| Adaptive voxel | O(N log N) | ~1.5 M pts/s |

GPU acceleration paths are deferred to [RFC-0006](RFC-0006-gpu-acceleration.md).

### Testing Plan

- `tests/test_downsample.py`:
  - FPS: output has exactly `num_points` points; same seed yields same result.
  - Poisson-disk: no two output points closer than `min_dist`.
  - Adaptive: output density approximately ≤ `target_density`.
  - Normal-based: requires normals; graceful error if normals absent.
  - Benchmark regression test in `tests/test_benchmark.py`.
- API docs: use context7 to check `rayon` parallel iterator API.

## Unresolved Questions

1. Should FPS support approximate variants (e.g., mini-batch FPS) for >10 M points?
2. For normal-based sampling, should curvature be precomputed and cached in the point cloud struct?
3. Adaptive voxel grid — should the octree structure be exposed as a queryable index (see RFC-0004 for registration use)?

## References

- Qi et al. (2017) "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" — FPS definition.
- Corsini et al. (2012) "Efficient and Flexible Sampling with Blue Noise Properties of Triangular Meshes" — Poisson-disk.
- [rayon crate](https://docs.rs/rayon)
- [RFC-0001: Core Architecture](RFC-0001-core-architecture.md)
- [RFC-0005: Normal Vector Estimation](RFC-0005-normal-estimation.md)
- [RFC-0006: GPU Acceleration Backend](RFC-0006-gpu-acceleration.md)
