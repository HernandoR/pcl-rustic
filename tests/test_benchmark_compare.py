"""Comparison benchmarks against Open3D and laspy.

Run with `just bench-compare` (installs the `bench` dependency group first).
Skipped automatically when the baselines are not installed.

Fairness notes -- read these before quoting any number:

* **Voxel downsample semantics differ.** Open3D's `voxel_down_sample`
  emits a synthetic averaged point per voxel; pcl-rustic's
  `NEAREST_TO_CENTROID` picks the nearest *existing* point, which costs an
  extra distance pass per voxel. The output counts printed alongside each
  timing make the difference visible. This is not an apples-to-apples
  kernel; it is an apples-to-apples *user task* ("downsample this cloud").
* **Transform result materialization differs.** Open3D transforms in place
  over f64 and `np.asarray(pcd.points)` is a zero-copy view, so its timing
  excludes any copy. pcl-rustic's ops are lazy, so the timed region must
  include reading `.xyz` back, which allocates an (N, 3) f8 array and
  converts from the f32 device tensor. pcl-rustic is therefore charged for
  work Open3D is not.
* Input construction is done outside every timed region.
* PCL (the C++ library) has no working Python binding on 3.12 -- `pclpy`
  pins `==3.6.*` and `python-pcl`'s last release is a 2018 cp27/cp35 alpha
  -- so it is not benchmarked here.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from pcl_rustic import DownsampleStrategy, PointCloud

o3d = pytest.importorskip("open3d", reason="open3d not installed")
laspy = pytest.importorskip("laspy", reason="laspy not installed")

pytestmark = [pytest.mark.bench, pytest.mark.slow]

POINT_COUNTS = [1_000_000, 10_000_000]
VOXEL_SIZE = 0.15
IO_POINTS = 2_000_000


def _gaussian_xyz(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=100.0, size=(n_points, 3))


def _timed(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


def _row(
    op: str, n: int, ours: float, theirs: float, baseline: str, extra: str = ""
) -> None:
    ratio = theirs / ours if ours > 0 else float("inf")
    verdict = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
    print(
        f"{op:<22} n={n:>11,}  pcl-rustic={ours:7.3f}s  "
        f"{baseline}={theirs:7.3f}s  -> {verdict}{extra}"
    )


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_compare_voxel_downsample(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)

    pc = PointCloud.from_xyz(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    ours, down = _timed(
        lambda: pc.voxel_downsample(
            VOXEL_SIZE, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
        )
    )
    theirs, down_o3d = _timed(lambda: pcd.voxel_down_sample(VOXEL_SIZE))

    _row(
        "voxel_downsample",
        n_points,
        ours,
        theirs,
        "open3d",
        f"  (out: {len(down):,} vs {len(down_o3d.points):,})",
    )
    assert 0 < len(down) <= n_points


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_compare_transform(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    pc = PointCloud.from_xyz(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    def ours_op() -> np.ndarray:
        # Lazy ops: reading .xyz back is what forces execution.
        return pc.transform(matrix).xyz

    def theirs_op() -> np.ndarray:
        return np.asarray(pcd.transform(matrix).points)

    ours, ours_result = _timed(ours_op)
    theirs, theirs_result = _timed(theirs_op)

    _row("transform (4x4)", n_points, ours, theirs, "open3d")
    # Both must agree on the actual math, or the comparison is meaningless.
    np.testing.assert_allclose(ours_result, theirs_result, atol=1e-3)


def test_compare_las_read(tmp_path: Path) -> None:
    xyz = _gaussian_xyz(IO_POINTS, seed=3)
    rng = np.random.default_rng(4)

    source = PointCloud.from_xyz(xyz)
    source["intensity"] = rng.integers(0, 65535, size=IO_POINTS).astype(np.uint16)
    source["gps_time"] = rng.normal(size=IO_POINTS) * 1.0e3
    path = tmp_path / "bench.las"
    source.write(str(path))

    def ours_op() -> np.ndarray:
        return PointCloud.read(str(path)).xyz

    def theirs_op() -> np.ndarray:
        las = laspy.read(str(path))
        return np.column_stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
        )

    ours, ours_xyz = _timed(ours_op)
    theirs, theirs_xyz = _timed(theirs_op)

    _row("LAS read + xyz", IO_POINTS, ours, theirs, "laspy ")
    # Coordinates are NOT bit-identical to laspy: ADR-0002 stores an f64
    # offset plus f32 relative coordinates, so reconstruction carries f32
    # epsilon at the cloud's relative extent. The contract is that this stays
    # below the LAS scale quantization (1e-3 by default), not that it is zero.
    las_scale = 1.0e-3
    max_dev = float(np.abs(ours_xyz - theirs_xyz).max())
    print(
        f"{'  coordinate fidelity':<22} max deviation vs laspy = {max_dev:.3e} m "
        f"({max_dev / las_scale:.1%} of the {las_scale} LAS scale step)"
    )
    assert max_dev < las_scale, (
        f"deviation {max_dev} exceeds LAS quantization {las_scale}"
    )


def test_compare_las_intensity_dtype(tmp_path: Path) -> None:
    """The dtype-promise claim, checked against laspy on a real file."""
    xyz = _gaussian_xyz(1000, seed=5)
    rng = np.random.default_rng(6)
    intensity = rng.integers(0, 65535, size=1000).astype(np.uint16)
    classification = rng.integers(0, 11, size=1000).astype(np.uint8)

    source = PointCloud.from_xyz(xyz)
    source["intensity"] = intensity
    source["classification"] = classification
    path = tmp_path / "dtype.las"
    source.write(str(path))

    ours = PointCloud.read(str(path))
    theirs = laspy.read(str(path))

    for dim in ("intensity", "classification"):
        mine = ours[dim]
        yours = np.asarray(getattr(theirs, dim))
        assert mine.dtype == yours.dtype, (
            f"{dim}: {mine.dtype} != laspy's {yours.dtype}"
        )
        np.testing.assert_array_equal(mine, yours)
    print(
        "dtype parity with laspy: intensity="
        f"{ours['intensity'].dtype}, classification={ours['classification'].dtype} (exact match)"
    )
