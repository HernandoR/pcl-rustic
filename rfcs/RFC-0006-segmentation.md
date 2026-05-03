# RFC-0006: Point Cloud Segmentation

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Draft                              |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2026-05-03                         |
| Updated | 2026-05-03                         |

## Summary

Add point-cloud segmentation algorithms to PCL Rustic. Phase 1 covers **RANSAC-based plane segmentation** and **Euclidean Cluster Extraction (connected components)**. Phase 2 adds **Region Growing** (normal-angle-based). These algorithms cover the majority of real-world segmentation use cases in robotics, autonomous driving, and geospatial applications.

## Motivation

Segmentation partitions a point cloud into semantically meaningful groups:

- **Plane segmentation** (RANSAC): extract ground plane, building façades, tabletops.
- **Cluster extraction**: identify individual objects (vehicles, pedestrians, trees).
- **Region growing**: delineate smooth surface patches (walls, roof panels).

Without segmentation, PCL Rustic cannot support common pre-processing pipelines such as ground removal before clustering or building extraction from LiDAR maps.

## Design Details

### Python API

```python
from pcl_rustic import SegmentationResult, PlaneModel

# RANSAC Plane Segmentation
seg_result = pc.segment_plane(
    distance_threshold=0.05,  # max point-plane distance to be an inlier
    ransac_n=3,               # points sampled per RANSAC iteration
    num_iterations=100,
)
plane_model: PlaneModel = seg_result.model     # ax + by + cz + d = 0
inlier_cloud: PointCloud = seg_result.inliers  # points on the plane
outlier_cloud: PointCloud = seg_result.outliers  # remaining points

# Euclidean Cluster Extraction
clusters: list[PointCloud] = pc.euclidean_cluster_extraction(
    eps=0.2,          # max distance between points in the same cluster
    min_points=10,    # minimum cluster size
    max_points=100_000,  # maximum cluster size (optional)
)

# Region Growing Segmentation (requires normals)
regions: list[PointCloud] = pc.region_growing(
    k_neighbors=30,
    smoothness_threshold=0.1,  # max angle (radians) between normals
    curvature_threshold=1.0,   # max curvature to seed a new region
    min_cluster_size=50,
)
```

### PlaneModel and SegmentationResult

```python
@dataclass
class PlaneModel:
    a: float  # normal x
    b: float  # normal y
    c: float  # normal z
    d: float  # offset
    # convenience
    normal: np.ndarray   # [a, b, c] shape (3,) float32
    inlier_ratio: float  # fraction of points on the plane

@dataclass
class SegmentationResult:
    model: PlaneModel
    inliers: PointCloud
    outliers: PointCloud
    num_inliers: int
```

Both implemented as PyO3 `#[pyclass]` types.

### RANSAC Plane Segmentation

Algorithm:
1. Randomly sample `ransac_n` (≥3) points.
2. Fit a plane through the sample.
3. Count inliers (points within `distance_threshold` of the plane).
4. Repeat `num_iterations` times; keep the plane with most inliers.
5. Refit plane to all inliers using least-squares.

Optional: **Progressive Sample Consensus (PROSAC)** ordering (deferred).

### Euclidean Cluster Extraction (DBSCAN-like)

Algorithm:
1. Build a spatial index (KD-Tree or voxel grid).
2. For each unvisited point, find neighbors within `eps`.
3. If ≥ `min_points` neighbors, start a new cluster and expand recursively (BFS).
4. Mark isolated points as noise (not returned in clusters list).

This is equivalent to DBSCAN with `min_samples=min_points` and `epsilon=eps`.

**Complexity**: O(N log N) with KD-Tree.

### Region Growing Segmentation

Prerequisite: normals must be pre-computed ([RFC-0005](./RFC-0005-normal-estimation.md)).

Algorithm:
1. Sort points by curvature (lowest first = seed candidates).
2. For each seed, expand region by adding neighbors whose normal angle to the region normal is < `smoothness_threshold`.
3. Stop when no more neighbors qualify or cluster reaches max size.

This produces smooth surface patches with consistent normals.

### Module Layout

```
src/
└── segmentation/
    ├── mod.rs            # re-exports
    ├── ransac_plane.rs   # RANSAC plane segmentation
    ├── euclidean.rs      # Euclidean cluster extraction (DBSCAN)
    ├── region_growing.rs # Region growing (depends on normals)
    └── result.rs         # SegmentationResult, PlaneModel PyO3 types
```

### GPU Acceleration (Phase 2)

RANSAC is embarrassingly parallel (each iteration is independent). With [RFC-0002](./RFC-0002-gpu-acceleration.md)'s GPU backend:
- Sample `num_iterations` triplets in parallel.
- Compute all plane models in a batch.
- Count inliers via Burn `lt` + `sum` in parallel.
- Select best plane via `argmax`.

Expected speedup: ~10–50× for `num_iterations` = 1000.

Euclidean cluster extraction is harder to parallelize; CPU KD-Tree remains the primary path.

### Performance Targets

| Algorithm                  | 10 M pts | 1 M pts  |
|----------------------------|----------|----------|
| RANSAC (100 iterations)    | <2 s     | <0.2 s   |
| Euclidean (CPU, KD-Tree)   | <10 s    | <1 s     |
| Region Growing (CPU)       | <15 s    | <1.5 s   |

## Alternatives Considered

### Alternative 1: Pure Python RANSAC + sklearn DBSCAN

Reject: too slow for 10 M+ points; adds Python dependencies.

### Alternative 2: HDBSCAN Instead of DBSCAN

HDBSCAN is more robust to varying cluster densities. Rejected for Phase 1 because:
- More complex to implement in Rust.
- DBSCAN meets 90% of use cases.
- HDBSCAN deferred to RFC-0011.

### Alternative 3: Deep Learning Segmentation (PointNet++)

Out of scope for a classical geometry library. To be addressed in a separate ML module RFC.

## Open Questions

1. **Noise points from Euclidean clustering**: Should noise points be returned as a separate `PointCloud` or silently discarded?
2. **RANSAC model variants**: Beyond planes, should this RFC include line, sphere, or cylinder RANSAC? (Recommend separate RFC-0012.)
3. **Shared KD-Tree with normals and registration**: Three RFCs (this one, RFC-0004, RFC-0005) all need a KD-Tree. Should the spatial index be a top-level module shared across all three?
4. **Multi-plane extraction**: Should `segment_plane()` be extended to extract multiple planes iteratively (subtract-and-repeat)?

## Implementation Plan

- [ ] Create `src/segmentation/` module
- [ ] Implement RANSAC plane in `src/segmentation/ransac_plane.rs`
- [ ] Implement Euclidean cluster extraction in `src/segmentation/euclidean.rs`
- [ ] Implement Region Growing in `src/segmentation/region_growing.rs` (after RFC-0005 normals)
- [ ] Create `SegmentationResult` and `PlaneModel` PyO3 classes
- [ ] Expose `segment_plane()`, `euclidean_cluster_extraction()`, `region_growing()` on `PointCloud`
- [ ] Add pytest tests: `tests/test_segmentation.py`
  - `test_ransac_flat_plane()` — perfect plane: 100% inlier ratio
  - `test_ransac_noisy_plane()` — noisy plane: inlier ratio > 90%
  - `test_ransac_plane_model_coefficients()` — model [a,b,c,d] matches ground truth
  - `test_euclidean_two_clusters()` — two separated spheres → exactly 2 clusters
  - `test_euclidean_min_points_filter()` — tiny cluster below min_points is excluded
  - `test_region_growing_requires_normals()` — raises if no normals
  - `test_region_growing_planar_patches()` — flat regions produce large smooth clusters
- [ ] Update `rfcs/README.md` index status

## Related RFCs

- **Background**: [RFC-0001](./RFC-0001-initial-architecture.md) — architecture and error handling
- **Accelerated by** ← [RFC-0002](./RFC-0002-gpu-acceleration.md): GPU RANSAC parallelism
- **Uses** ← [RFC-0005](./RFC-0005-normal-estimation.md): normals required for Region Growing
- **Shares spatial index with** ← [RFC-0004](./RFC-0004-point-cloud-registration.md): KD-Tree module
- **Future** → RFC-0011 (planned): HDBSCAN clustering
- **Future** → RFC-0012 (planned): geometric model fitting (cylinder, sphere, line)

## References

- Fischler & Bolles "Random Sample Consensus: A Paradigm for Model Fitting" (CACM 1981)
- Ester et al. "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases" (KDD 1996) — DBSCAN
- Rabbani et al. "Segmentation of Point Clouds Using Smoothness Constraint" (ISPRS 2006) — Region Growing
- [PCL Segmentation tutorials](https://pointclouds.org/documentation/tutorials/planar_segmentation.html)
- [Open3D cluster_dbscan](http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.cluster_dbscan)
- [RFC-0001](./RFC-0001-initial-architecture.md)
- [RFC-0002](./RFC-0002-gpu-acceleration.md)
- [RFC-0004](./RFC-0004-point-cloud-registration.md)
- [RFC-0005](./RFC-0005-normal-estimation.md)
