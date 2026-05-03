# RFC-0006 — Point Cloud Segmentation

| Field | Value |
|-------|-------|
| RFC | 0006 |
| Status | Open for Review |
| Author | liuzhen19 |
| Created | 2026-05-03 |
| Updated | 2026-05-03 |

## Summary

Add point cloud segmentation algorithms to PCL Rustic, starting with **RANSAC plane segmentation** and **Euclidean cluster extraction**. A follow-up RFC will add normal-based region growing. This RFC defines the segmentation trait, result types, and Python API.

## Motivation

Segmentation is one of the most common point cloud operations in robotics and geospatial workflows:

- **Ground plane removal** — detect and remove the dominant plane (ground, floor) using RANSAC before object detection.
- **Object clustering** — group spatially connected points into individual objects (trees, vehicles, buildings).
- **Region growing** — group points with similar normals for surface segmentation (patches, walls, roofs).

These three operations cover > 80% of real-world segmentation needs. Without native support, users write slow Python loops or call Open3D, which requires a heavy C++ install.

## Design

### Result types

```rust
// src/traits/segmentation.rs

/// A labelled set of point indices.
pub struct Segment {
    pub label:   i32,          // -1 = outlier / unclassified
    pub indices: Vec<usize>,
}

/// Output of a segmentation algorithm.
pub struct SegmentationResult {
    pub segments: Vec<Segment>,
    pub labels:   Vec<i32>,   // length N; labels[i] = segment label for point i
}
```

### Segmentation trait

```rust
pub trait PointCloudSegmentation {
    fn segment(
        &self,
        config: &SegmentationConfig,
    ) -> Result<SegmentationResult, PointCloudError>;
}

pub enum SegmentationAlgorithm {
    RansacPlane,
    EuclideanClustering,
    RegionGrowing,  // requires normals (RFC-0005)
}

pub struct SegmentationConfig {
    pub algorithm: SegmentationAlgorithm,

    // RANSAC parameters
    pub distance_threshold: f32,   // inlier distance (default: 0.02)
    pub max_iterations:     usize, // default: 1000
    pub probability:        f32,   // success probability (default: 0.99)

    // Euclidean clustering parameters
    pub cluster_tolerance:  f32,   // default: 0.02
    pub min_cluster_size:   usize, // default: 100
    pub max_cluster_size:   usize, // default: 25_000

    // Region growing parameters (requires normals)
    pub smooth_threshold:   f32,   // normal angle threshold in radians (default: 0.26)
    pub curvature_threshold: f32,  // max curvature (default: 1.0)
    pub k_neighbours:       usize, // default: 30
}
```

### RANSAC plane segmentation

1. Sample 3 random points to define a plane hypothesis.
2. Count inliers (points within `distance_threshold` of the plane).
3. Repeat `max_iterations` times; keep the best hypothesis.
4. Returns two segments: **inliers** (the plane, label `0`) and **outliers** (label `-1`).

The minimum number of iterations for confidence `p` with inlier ratio `w`:

```
n = ceil(log(1 - p) / log(1 - w^3))
```

### Euclidean cluster extraction

1. Build a k-d tree over all points.
2. Seed from an unvisited point; BFS/DFS expanding to neighbours within `cluster_tolerance`.
3. Accept cluster if its size is within `[min_cluster_size, max_cluster_size]`.
4. Label each accepted cluster `0, 1, 2, …`; unclustered points get label `-1`.

### Region growing (basic implementation)

1. Sort points by curvature (computed from normals; see RFC-0005).
2. Seed from the lowest-curvature point; grow by checking both:
   - Euclidean distance < `cluster_tolerance`
   - Angular difference between normals < `smooth_threshold`
3. Accept or split segments based on `curvature_threshold`.

Requires normals to be pre-computed. Returns error if `has_normals()` is false.

### Python API

```python
from pcl_rustic import PointCloud, SegmentationConfig, SegmentationAlgorithm

# Ground plane removal (RANSAC)
config = SegmentationConfig(
    algorithm=SegmentationAlgorithm.RANSAC_PLANE,
    distance_threshold=0.02,
    max_iterations=1000,
)
result = pc.segment(config)

plane_indices = result.segments[0].indices       # inliers
objects_pc = pc.select_by_indices(              # remaining points
    result.segments[1].indices
)

# Euclidean clustering
config = SegmentationConfig(
    algorithm=SegmentationAlgorithm.EUCLIDEAN_CLUSTERING,
    cluster_tolerance=0.5,
    min_cluster_size=50,
    max_cluster_size=10_000,
)
result = pc.segment(config)
print(f"Found {len(result.segments)} clusters")

for seg in result.segments:
    if seg.label >= 0:
        cluster_pc = pc.select_by_indices(seg.indices)
        print(f"  Cluster {seg.label}: {cluster_pc.point_count()} points")

# Get per-point labels as numpy array
labels = result.labels  # np.ndarray of int32, shape (N,)
```

### New `select_by_indices` method

A utility method is needed to extract a sub-cloud by index list:

```rust
// On HighPerformancePointCloud
pub fn select_by_indices(&self, indices: &[usize]) -> Result<Self, PointCloudError>;
```

This is also useful for downsampling result inspection and other operations.

### Justfile additions

```just
# Run segmentation tests
test-segmentation:
    uv run pytest tests/test_segmentation.py -v

# Run slow segmentation benchmarks
benchmark-segmentation:
    uv run pytest tests/test_benchmark.py -k segmentation -v -s
```

### Integration tests

```python
# tests/test_segmentation.py
import numpy as np
import pytest
from pcl_rustic import PointCloud, SegmentationConfig, SegmentationAlgorithm

@pytest.fixture
def plane_with_noise():
    """A flat plane at z=0 with scattered noise points above."""
    rng = np.random.default_rng(99)
    plane_pts = rng.uniform(-5, 5, (900, 2)).astype(np.float32)
    plane_z = np.zeros((900, 1), dtype=np.float32)
    plane = np.concatenate([plane_pts, plane_z], axis=1)

    noise_pts = rng.uniform(-5, 5, (100, 3)).astype(np.float32)
    noise_pts[:, 2] += rng.uniform(0.5, 2.0, 100).astype(np.float32)

    xyz = np.concatenate([plane, noise_pts], axis=0)
    return PointCloud.from_xyz(xyz)

@pytest.fixture
def two_clusters():
    """Two well-separated spherical clusters."""
    rng = np.random.default_rng(42)
    c1 = rng.standard_normal((200, 3)).astype(np.float32) * 0.3
    c2 = rng.standard_normal((200, 3)).astype(np.float32) * 0.3 + 5.0
    xyz = np.concatenate([c1, c2], axis=0)
    return PointCloud.from_xyz(xyz)

def test_ransac_finds_plane(plane_with_noise):
    config = SegmentationConfig(
        algorithm=SegmentationAlgorithm.RANSAC_PLANE,
        distance_threshold=0.05,
    )
    result = plane_with_noise.segment(config)
    plane_seg = next(s for s in result.segments if s.label == 0)
    # Should recover most of the 900 plane points
    assert len(plane_seg.indices) >= 800

def test_euclidean_finds_two_clusters(two_clusters):
    config = SegmentationConfig(
        algorithm=SegmentationAlgorithm.EUCLIDEAN_CLUSTERING,
        cluster_tolerance=1.0,
        min_cluster_size=50,
        max_cluster_size=500,
    )
    result = two_clusters.segment(config)
    valid = [s for s in result.segments if s.label >= 0]
    assert len(valid) == 2

def test_select_by_indices(two_clusters):
    config = SegmentationConfig(
        algorithm=SegmentationAlgorithm.EUCLIDEAN_CLUSTERING,
        cluster_tolerance=1.0,
        min_cluster_size=50,
    )
    result = two_clusters.segment(config)
    seg0 = next(s for s in result.segments if s.label == 0)
    sub = two_clusters.select_by_indices(seg0.indices)
    assert sub.point_count() == len(seg0.indices)
```

## Drawbacks

- RANSAC is probabilistic; results vary between runs (but are reproducible with a seeded RNG).
- Euclidean clustering via BFS can be slow for dense clouds with many small clusters (many BFS queue operations).
- Region growing requires normals (RFC-0005); cannot be tested until RFC-0005 is implemented.
- `select_by_indices` copies data; no zero-copy slice view possible with current storage layout.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| Min-cut / graph-cut segmentation | Requires graph construction; complex; defer to future RFC |
| Supervoxel clustering (VCCS) | Complex to implement; better suited to coloured point clouds |
| DBSCAN | Similar to Euclidean clustering but density-adaptive; could be added later |
| Conditional Euclidean Clustering | Extension of Euclidean with attribute conditioning; follow-up RFC |

## Unresolved Questions

1. Should `SegmentationResult.labels` be exposed as a `numpy` array of `int32` or `int64`?
2. How to seed the RANSAC random number generator? Global or per-call seed parameter?
3. Should `select_by_indices` preserve attribute channels that the sub-cloud's points don't fully cover?
4. What is the performance target for RANSAC on 1 M points?
5. Should region-growing be deferred to RFC-0008 to keep this RFC focused?

## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — `PointCloudCore` and attribute channels underpin this RFC
- [RFC-0004 — Point Cloud Registration](./RFC-0004-point-cloud-registration.md) — often applied after plane removal; shares k-d tree infrastructure
- [RFC-0005 — Normal Estimation](./RFC-0005-normal-estimation.md) — region-growing segmentation requires normals
- [RFC-0007 — Testing & CI Infrastructure](./RFC-0007-testing-ci-infrastructure.md) — slow-test marker for probabilistic tests

## Reviews

### Review 1 — Sub-agent Alpha

> **Status**: APPROVED
>
> The design is comprehensive. RANSAC plane and Euclidean clustering are the correct first two algorithms. The `select_by_indices` utility method is a necessary addition and should be in the public Python API (it is broadly useful). The `SegmentationConfig` struct is well-structured but consider whether a builder pattern would be more ergonomic for Python users (keyword arguments already handle this via PyO3). Recommend resolving Unresolved Question 5 (defer region growing to RFC-0008) to keep this RFC achievable — region growing's dependency on RFC-0005 adds risk.

### Review 2 — Sub-agent Beta

> **Status**: APPROVED (with minor comments)
>
> RANSAC implementation description is correct. The probability formula for minimum iterations is standard and should be implemented precisely. One concern: the BFS for Euclidean clustering on a 50 M point cloud could exhaust the call stack; use an explicit queue (`VecDeque`), not recursion. The integration tests are deterministic (fixed seeds), which is good for CI. Recommend adding a performance regression test to the benchmark suite (e.g., assert RANSAC on 100 K points completes in < 2 s).
