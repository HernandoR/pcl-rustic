# RFC-0007: Point Cloud Segmentation & Clustering

- **Status**: Draft
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-05-03
- **Updated**: 2026-05-03

---

## Summary

Add point cloud segmentation and clustering capabilities to pcl-rustic, including DBSCAN, Euclidean Cluster Extraction, RANSAC-based geometric primitive fitting (plane, sphere, cylinder), and region growing.  
Results are returned as label arrays, enabling downstream per-cluster processing.

---

## Motivation

Segmentation unlocks a wide family of robotics and geomatics workflows:
- **Ground removal** (RANSAC plane) before building/object detection
- **Object clustering** (DBSCAN / Euclidean) for bounding box extraction
- **Structural analysis** (cylinder fitting for pipe inspection, sphere fitting for target detection)
- **Region growing** for smooth-surface extraction in AEC scans

These algorithms all have Python-side implementations in Open3D and python-pcl, but they are slow on large clouds.  
A native Rust implementation with `rayon` parallelism and KD-tree from `kiddo` (RFC-0005/RFC-0006) will be significantly faster.

---

## Detailed Design

### Unified Segmentation API

All segmentation methods return a `SegmentationResult`:

```python
from pcl_rustic import PointCloud, SegmentationResult

result: SegmentationResult = pc.segment(method, **kwargs)

labels: np.ndarray = result.labels    # shape (N,), dtype int32
                                      # -1 = noise/outlier
n_clusters: int = result.n_clusters
```

Labels can be used to extract sub-clouds:

```python
cluster_0 = pc.select_by_label(result.labels, label=0)
```

### Segmentation Methods

#### DBSCAN

```python
result = pc.segment_dbscan(
    eps: float,          # neighbourhood radius
    min_samples: int,    # minimum points to form a cluster
) -> SegmentationResult
```

- Uses `kiddo` radius search for neighbourhood queries
- Parallelised with `rayon`
- Returns `-1` for noise points

#### Euclidean Cluster Extraction

```python
result = pc.segment_euclidean(
    cluster_tolerance: float,
    min_cluster_size: int = 100,
    max_cluster_size: int = 25_000,
) -> SegmentationResult
```

Similar to PCL's `EuclideanClusterExtraction`; uses KD-tree radius search + BFS labelling.

#### RANSAC Plane Fitting

```python
from pcl_rustic import RansacModel

result = pc.fit_ransac(
    model: RansacModel = RansacModel.PLANE,
    *,
    distance_threshold: float = 0.01,
    max_iterations: int = 1000,
    probability: float = 0.99,
) -> RansacResult

# RansacResult fields:
result.inlier_mask     # np.ndarray (N,) bool — inlier/outlier mask
result.model_coeffs    # np.ndarray (4,) for PLANE [a,b,c,d]: ax+by+cz+d=0
                       # np.ndarray (7,) for SPHERE [cx,cy,cz,r,nx,ny,nz]
                       # np.ndarray (7,) for CYLINDER
result.n_inliers       # int
```

#### Region Growing

```python
result = pc.segment_region_growing(
    k_neighbours: int = 30,
    smoothness_threshold: float = 0.1,  # radians
    curvature_threshold: float = 1.0,
    min_cluster_size: int = 50,
) -> SegmentationResult
```

Requires normals to be present (RFC-0006).  
Seeds from lowest-curvature points; grows by angular normal deviation criterion.

### `RansacModel` Enum

```python
class RansacModel(IntEnum):
    PLANE = 0
    SPHERE = 1
    CYLINDER = 2
```

### `select_by_label` Helper

```python
cluster_cloud: PointCloud = pc.select_by_label(labels: np.ndarray, label: int) -> PointCloud
```

Returns a new `PointCloud` containing only points where `labels == label`.

---

## Implementation Plan

- [ ] Add `src/segmentation/mod.rs`
- [ ] Implement `src/segmentation/dbscan.rs`
- [ ] Implement `src/segmentation/euclidean.rs`
- [ ] Implement `src/segmentation/ransac.rs` (plane, sphere, cylinder)
- [ ] Implement `src/segmentation/region_growing.rs`
- [ ] Add `SegmentationResult` and `RansacResult` as PyO3 classes
- [ ] Expose `segment_dbscan`, `segment_euclidean`, `fit_ransac`, `segment_region_growing`, `select_by_label` in `src/lib.rs`
- [ ] Add `tests/test_segmentation.py`
- [ ] Benchmark DBSCAN and RANSAC plane on 1M points; add to `tests/test_benchmark.py`
- [ ] Update `docs/api/` with new `segmentation.md`

---

## Testing Strategy

```python
# tests/test_segmentation.py
import numpy as np
import pytest
from pcl_rustic import PointCloud, RansacModel

def make_two_clusters(n=200):
    """Two spatially separated Gaussian clusters."""
    c1 = np.random.randn(n, 3).astype(np.float32) + np.array([0, 0, 0])
    c2 = np.random.randn(n, 3).astype(np.float32) + np.array([10, 10, 10])
    xyz = np.vstack([c1, c2])
    return PointCloud.from_xyz(xyz)

def test_dbscan_two_clusters():
    pc = make_two_clusters(200)
    result = pc.segment_dbscan(eps=1.5, min_samples=5)
    assert result.n_clusters == 2
    # No noise
    assert (result.labels >= 0).all()

def test_dbscan_noise_label():
    # A single isolated point should be labelled -1
    isolated = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    cluster = np.random.randn(100, 3).astype(np.float32)
    pc = PointCloud.from_xyz(np.vstack([cluster, isolated]))
    result = pc.segment_dbscan(eps=0.5, min_samples=5)
    assert -1 in result.labels

def test_euclidean_cluster_count():
    pc = make_two_clusters(500)
    result = pc.segment_euclidean(cluster_tolerance=1.5, min_cluster_size=10)
    assert result.n_clusters == 2

def test_ransac_plane():
    # Create a noisy plane z=0
    xy = np.random.uniform(-5, 5, (900, 2)).astype(np.float32)
    z = (np.random.randn(900) * 0.005).astype(np.float32).reshape(-1, 1)
    plane_pts = np.hstack([xy, z])
    # Add noise points
    noise = np.random.uniform(-5, 5, (100, 3)).astype(np.float32)
    noise[:, 2] += 2.0  # lift noise off the plane
    xyz = np.vstack([plane_pts, noise])
    pc = PointCloud.from_xyz(xyz)
    result = pc.fit_ransac(RansacModel.PLANE, distance_threshold=0.05)
    assert result.n_inliers >= 850  # most plane points should be inliers

def test_select_by_label():
    pc = make_two_clusters(200)
    result = pc.segment_dbscan(eps=1.5, min_samples=5)
    cluster_0 = pc.select_by_label(result.labels, label=0)
    cluster_1 = pc.select_by_label(result.labels, label=1)
    assert cluster_0.point_count() + cluster_1.point_count() == 400

def test_region_growing_requires_normals():
    xyz = np.random.randn(100, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    with pytest.raises(ValueError, match="normals"):
        pc.segment_region_growing(k_neighbours=10)

def test_region_growing_with_normals():
    xyz = np.random.randn(500, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc.estimate_normals(k=20)
    result = pc.segment_region_growing(k_neighbours=20, min_cluster_size=5)
    assert result.n_clusters >= 1

@pytest.mark.slow
def test_dbscan_1m():
    xyz = np.random.randn(1_000_000, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    result = pc.segment_dbscan(eps=0.1, min_samples=5)
    assert result.n_clusters >= 0
```

---

## Alternatives Considered

- **scikit-learn DBSCAN via Python**: O(N²) memory, single-threaded, not usable for N > 100K.
- **Open3D segmentation**: heavyweight dependency.
- **Only implement RANSAC plane (most common need)**: insufficient; DBSCAN and region growing are equally important.

---

## Drawbacks

- RANSAC is non-deterministic; tests must allow tolerance ranges.
- Region growing is sensitive to normal quality (RFC-0006) and threshold tuning.
- DBSCAN worst-case is O(N²) without spatial indexing; `kiddo` radius search brings it to O(N log N) average.

---

## Unresolved Questions

- Should `fit_ransac` support iterative plane removal (segment all planes in order)?
- Should `SegmentationResult.labels` be `int32` or `int64` for >2B-point clouds?
- Should cylinder RANSAC require pre-estimated normals, or estimate them internally?
- Should `select_by_label` support a list of labels (multi-label select)?

---

## Related RFCs

- [RFC-0001](./RFC-0001-foundation.md) — PointCloudProperties trait; select_by_label returns new PointCloud
- [RFC-0002](./RFC-0002-tensor-backend.md) — GPU-accelerated parallel DBSCAN distance queries
- [RFC-0006](./RFC-0006-normal-estimation.md) — Region growing requires normals; RANSAC cylinder benefits from them
