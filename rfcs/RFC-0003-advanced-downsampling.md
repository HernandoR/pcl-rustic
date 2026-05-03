# RFC-0003 — Advanced Downsampling Strategies (FPS & Normal-based)

| Field | Value |
|-------|-------|
| RFC | 0003 |
| Status | Open for Review |
| Author | liuzhen19 |
| Created | 2026-05-03 |
| Updated | 2026-05-03 |

## Summary

Add two new downsampling strategies — **Farthest Point Sampling (FPS)** and **Normal-based Sampling** — to complement the existing `RANDOM` and `CENTROID` voxel strategies. Both strategies are commonly used in deep-learning point cloud pipelines (PointNet++, etc.) and require exposing them at both the Rust trait level and the Python API.

## Motivation

The existing `RANDOM` and `CENTROID` strategies are fast but geometrically unaware:

- **RANDOM** does not guarantee spatial coverage. Dense regions dominate the sample.
- **CENTROID** preserves local density but still discards uniform surface coverage.

For deep-learning preprocessing and surface reconstruction workflows, two additional strategies are essential:

1. **FPS** — greedily selects the point farthest from all already-selected points. Guarantees spatially uniform coverage of the input point cloud. Standard in PointNet++, VoxelNet, and many 3D detection frameworks.
2. **Normal-based** — preferentially retains points near high-curvature regions (edges, corners). Requires normals (see RFC-0005) to be pre-computed.

## Design

### Trait extension

The existing `DownsampleStrategy` trait is extended with an optional context parameter:

```rust
// src/traits/downsample.rs
pub trait DownsampleStrategy: Send + Sync {
    /// Select one representative index from `candidate_indices`.
    ///
    /// `context` provides read-only access to all point attributes.
    fn select_representative(
        &self,
        candidate_indices: &[usize],
        context: &DownsampleContext<'_>,
    ) -> Result<usize, PointCloudError>;

    fn name(&self) -> &str;
}

pub struct DownsampleContext<'a> {
    pub xyz:       &'a [Vec<f32>],       // [N, 3]
    pub normals:   Option<&'a [Vec<f32>]>, // [N, 3] — populated if available
    pub intensity: Option<&'a [f32]>,
}
```

This is a **breaking change** to the existing trait signature. Existing `RandomSampleStrategy` and `CentroidSampleStrategy` must be updated to accept `DownsampleContext`.

### FPS strategy

```rust
// src/point_cloud/voxel_fps.rs
pub struct FarthestPointStrategy {
    num_samples: usize,
}

impl FarthestPointStrategy {
    pub fn new(num_samples: usize) -> Self { Self { num_samples } }
}
```

FPS operates on the **entire point cloud**, not per-voxel, so a new top-level method is introduced:

```rust
// On HighPerformancePointCloud
pub fn fps_downsample(&self, num_samples: usize) -> Result<Self, PointCloudError>;
```

The algorithm:

1. Pick a random seed point.
2. Maintain a `min_dist: Vec<f32>` of length N, initialised to `f32::MAX`.
3. For each iteration, select `argmax(min_dist)` as the next sample.
4. Update `min_dist` with distances to the newly selected point.

Time complexity: O(N × K) where K = `num_samples`. For large N and K, GPU acceleration (RFC-0002) is recommended.

### Normal-based sampling

```rust
// src/point_cloud/voxel_normal.rs
pub struct NormalBasedStrategy {
    curvature_weight: f32, // 0.0 = ignore curvature, 1.0 = full curvature weight
}
```

Curvature proxy: for each point, compute the variance of neighbouring normals. Points with higher variance lie near edges or surface discontinuities. The strategy selects the point with the highest curvature proxy within each voxel.

**Prerequisite**: normals must be pre-computed (see RFC-0005). If normals are absent, falls back to `CENTROID`.

### Python API

```python
from pcl_rustic import PointCloud, DownsampleStrategy

# Existing strategies (backward compatible)
pc.voxel_downsample(voxel_size=0.1, strategy=DownsampleStrategy.RANDOM)
pc.voxel_downsample(voxel_size=0.1, strategy=DownsampleStrategy.CENTROID)

# New: FPS (whole-cloud, not voxel-based)
pc_fps = pc.fps_downsample(num_samples=1024)

# New: normal-based voxel downsampling
pc_normal = pc.voxel_downsample(
    voxel_size=0.1,
    strategy=DownsampleStrategy.NORMAL_BASED,
    curvature_weight=0.8,
)
```

`DownsampleStrategy` gains two new variants:

```python
class DownsampleStrategy(IntEnum):
    RANDOM       = 0
    CENTROID     = 1
    NORMAL_BASED = 2   # new
    # FPS is method-level (fps_downsample), not a voxel strategy
```

### Justfile additions

```just
# Benchmark FPS vs voxel downsampling
benchmark-fps:
    uv run pytest tests/test_benchmark.py -k fps -v -s
```

### Integration tests

```python
# tests/test_downsampling.py
import numpy as np
import pytest
from pcl_rustic import PointCloud, DownsampleStrategy

@pytest.fixture
def large_cloud():
    rng = np.random.default_rng(42)
    xyz = rng.standard_normal((10_000, 3)).astype(np.float32)
    return PointCloud.from_xyz(xyz)

def test_fps_returns_correct_count(large_cloud):
    result = large_cloud.fps_downsample(num_samples=512)
    assert result.point_count() == 512

def test_fps_spatial_coverage(large_cloud):
    """FPS samples should cover the bounding box more uniformly than random."""
    fps = large_cloud.fps_downsample(num_samples=512).get_xyz()
    rnd = large_cloud.voxel_downsample(
        voxel_size=0.5, strategy=DownsampleStrategy.RANDOM
    ).get_xyz()
    # Std dev of FPS sample coordinates should be ≥ random's (more spread)
    assert np.std(fps) >= np.std(rnd) * 0.9

def test_normal_based_fallback_without_normals(large_cloud):
    """Without normals, NORMAL_BASED should fall back to CENTROID."""
    result = large_cloud.voxel_downsample(
        voxel_size=0.5, strategy=DownsampleStrategy.NORMAL_BASED
    )
    assert result.point_count() > 0
```

## Drawbacks

- FPS is O(N × K) — for 50 M points and 10 K samples it is expensive on CPU. GPU backend (RFC-0002) is recommended.
- `DownsampleContext` is a breaking change to the internal `DownsampleStrategy` trait (Rust-private; does not break the Python API).
- Normal-based sampling requires RFC-0005 to be useful; until then, it silently degrades.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| Poisson disk sampling | O(N log N); harder to implement on GPU; less common in ML pipelines |
| Grid sampling (already have) | Voxel downsample already covers this |
| Blue-noise sampling | Excellent quality but complex implementation; can be added later |

## Unresolved Questions

1. Should FPS accept a `seed` parameter for reproducibility?
2. Should `fps_downsample` preserve all attribute channels, or only XYZ?
3. What is the performance target for FPS on 1 M points on CPU?
4. Should `NORMAL_BASED` warn (log) when falling back to `CENTROID`?

## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — defines the `DownsampleStrategy` trait extended here
- [RFC-0002 — GPU Acceleration](./RFC-0002-gpu-acceleration.md) — GPU backend is needed for FPS at scale
- [RFC-0005 — Normal Estimation](./RFC-0005-normal-estimation.md) — `NORMAL_BASED` strategy depends on normals from this RFC
- [RFC-0007 — Testing & CI Infrastructure](./RFC-0007-testing-ci-infrastructure.md) — benchmark and test conventions

## Reviews

### Review 1 — Sub-agent Alpha

> **Status**: APPROVED
>
> The FPS algorithm description is correct. Using `argmax(min_dist)` is the canonical greedy approach and is well-suited for a Rust implementation. The `DownsampleContext` refactor is a good design decision — it makes the trait more composable without breaking the Python API. Recommend adding a section on expected memory usage for the `min_dist` buffer (N × 4 bytes — trivial compared to the point cloud itself). Unresolved Question 1 (seed) should be resolved before acceptance; reproducibility is important for ML pipelines.

### Review 2 — Sub-agent Beta

> **Status**: APPROVED (with minor comments)
>
> The fallback behaviour for `NORMAL_BASED` without normals is sensible. The integration test `test_fps_spatial_coverage` is a good property-based check. One concern: `std::std` is not a reliable measure of spatial coverage — consider using a bounding-box volume or a minimum spanning tree metric instead. The justfile recipe is straightforward. Suggest clarifying in the API docs that `fps_downsample` does not accept a `voxel_size` argument (to avoid user confusion with `voxel_downsample`).
