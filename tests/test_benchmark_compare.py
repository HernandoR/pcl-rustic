"""Comparison benchmarks: pcl-rustic vs Open3D, PCL, and laspy.

Run with `just bench-compare` (Open3D + laspy) or `just bench-pcl` (adds the
PCL baseline). Every baseline is optional; missing ones skip.

Baseline sourcing
-----------------
Open3D and laspy are ordinary Python packages in the `test` dependency group.
PCL is **not**, because it has no working Python binding on modern Python:
`python-pcl`'s newest release (0.3.0a1, 2018) ships only cp27/cp35 wheels and
its `setup.py` hard-codes PCL 1.7/1.8/1.9 branches -- against PCL 1.14 it
falls through to `else: pass` and dies with
`NameError: name 'vtk_include_dir' is not defined`. `pclpy` pins
`requires_python == 3.6.*` with a Windows-only wheel. The PCL baseline is
therefore a small C++ harness (`bench/pcl_cpp/`) invoked from this module, so
the comparison is still driven and reported from Python.

Fairness notes -- read these before quoting any number
------------------------------------------------------
* All engines consume byte-identical input: the C++ harness reads the same
  f64 xyz dump written from the very array handed to Open3D and pcl-rustic.
* Cloud construction is outside every timed region, on all sides.
* **Voxel semantics differ.** Open3D's `voxel_down_sample` and PCL's
  `VoxelGrid` emit a synthetic averaged point per voxel; pcl-rustic's
  `NEAREST_TO_CENTROID` picks the nearest *existing* point, which costs an
  extra distance pass. Output counts are printed so the difference is
  visible. This compares a user task, not an identical kernel.
* **Transform result materialization differs.** Open3D transforms in place
  over f64 and `np.asarray(pcd.points)` is a zero-copy view; PCL writes into
  a second f32 cloud. pcl-rustic's ops are lazy, so its timed region must
  include reading `.xyz` back, which allocates an (N, 3) f8 array and
  converts from the f32 device tensor. pcl-rustic is charged for work the
  others are not.
* **Element types differ by design.** PCL `PointXYZ` is f32, Open3D is f64,
  pcl-rustic is f32 relative to an f64 offset.
"""

from __future__ import annotations

import statistics
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from pcl_rustic import DownsampleStrategy, PointCloud

o3d = pytest.importorskip("open3d", reason="open3d not installed")
laspy = pytest.importorskip("laspy", reason="laspy not installed")

pytestmark = [pytest.mark.bench, pytest.mark.slow]

PCL_BIN = (
    Path(__file__).resolve().parents[1] / "bench" / "pcl_cpp" / "build" / "pcl_bench"
)

POINT_COUNTS = [1_000_000, 10_000_000]
VOXEL_SIZE = 0.15
# PCL's VoxelGrid overflows its int32 voxel index at VOXEL_SIZE over this
# extent, so the three-way run uses a leaf it accepts. The refusal is
# exercised on its own below.
PCL_SAFE_LEAF = 1.0
PCL_POINTS = 2_000_000
IO_POINTS = 2_000_000


def _gaussian_xyz(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=100.0, size=(n_points, 3))


#: One warmup run, then the median of this many timed runs. Without the
#: warmup, pcl-rustic's first op in a process pays a one-time kernel-init
#: cost (~55 ms), which made results depend on test ordering.
REPEAT = 3


def _timed(fn, repeat: int = REPEAT):
    fn()  # warmup
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result


def _verdict(ours: float, theirs: float) -> str:
    ratio = theirs / ours if ours > 0 else float("inf")
    return f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"


def _row(
    op: str, n: int, ours: float, theirs: float, baseline: str, extra: str = ""
) -> None:
    print(
        f"{op:<26} n={n:>11,}  pcl-rustic={ours:7.3f}s  {baseline}={theirs:7.3f}s"
        f"  -> pcl-rustic is {_verdict(ours, theirs)}{extra}"
    )


def _o3d_cloud(xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def _transform_matrix() -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    return matrix


# -- PCL C++ harness bridge ---------------------------------------------


def _require_pcl() -> None:
    if not PCL_BIN.exists():
        pytest.skip(f"PCL harness not built ({PCL_BIN}); run `just bench-pcl`")


def _dump_xyz(xyz: np.ndarray, path: Path) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<q", xyz.shape[0]))
        fh.write(np.ascontiguousarray(xyz, dtype="<f8").tobytes())


def _run_pcl(path: Path, leaf: float) -> dict[str, float]:
    proc = subprocess.run(
        [str(PCL_BIN), str(path), str(leaf), str(REPEAT)],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = dict(tok.split("=", 1) for tok in proc.stdout.split())
    return {k: float(v) for k, v in fields.items()}


# -- Open3D comparisons --------------------------------------------------


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_vs_open3d_voxel_downsample(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    pc = PointCloud.from_xyz(xyz)
    pcd = _o3d_cloud(xyz)

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
def test_vs_open3d_transform(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    matrix = _transform_matrix()
    pc = PointCloud.from_xyz(xyz)
    pcd = _o3d_cloud(xyz)

    # Open3D's transform mutates in place, so repeated calls compose. The
    # per-call work is identical either way, so the timing stays valid, but
    # the numerical check has to run on untouched clouds (below).
    ours, _ = _timed(lambda: pc.transform(matrix).xyz)
    theirs, _ = _timed(lambda: np.asarray(pcd.transform(matrix).points))

    _row("transform (4x4)", n_points, ours, theirs, "open3d")

    # Both must agree on the math, or the comparison is meaningless: one
    # application each, on fresh clouds.
    fresh_ours = PointCloud.from_xyz(xyz).transform(matrix).xyz
    fresh_theirs = np.asarray(_o3d_cloud(xyz).transform(matrix).points)
    np.testing.assert_allclose(fresh_ours, fresh_theirs, atol=1e-3)


# -- PCL comparisons -----------------------------------------------------


def test_vs_pcl_cpp(tmp_path: Path) -> None:
    _require_pcl()
    xyz = _gaussian_xyz(PCL_POINTS)
    data = tmp_path / "cloud.bin"
    _dump_xyz(xyz, data)

    pcl = _run_pcl(data, PCL_SAFE_LEAF)
    assert not pcl["voxel_refused"], "PCL refused the leaf size; comparison invalid"

    pc = PointCloud.from_xyz(xyz)
    ours_voxel, down = _timed(
        lambda: pc.voxel_downsample(
            PCL_SAFE_LEAF, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
        )
    )
    ours_transform, moved = _timed(lambda: pc.transform(_transform_matrix()).xyz)

    _row(
        f"voxel_downsample (leaf {PCL_SAFE_LEAF})",
        PCL_POINTS,
        ours_voxel,
        pcl["voxel_secs"],
        "pcl-c++",
        f"  (out: {len(down):,} vs {int(pcl['voxel_out']):,})",
    )
    _row(
        "transform (4x4)", PCL_POINTS, ours_transform, pcl["transform_secs"], "pcl-c++"
    )
    # Same voxel binning decision as PCL, independent of speed.
    assert len(down) == int(pcl["voxel_out"])
    assert moved.shape == (PCL_POINTS, 3)


def test_pcl_voxelgrid_refuses_fine_leaf(tmp_path: Path) -> None:
    """PCL silently no-ops where pcl-rustic downsamples; keep that documented."""
    _require_pcl()
    xyz = _gaussian_xyz(PCL_POINTS)
    data = tmp_path / "cloud.bin"
    _dump_xyz(xyz, data)

    pcl = _run_pcl(data, VOXEL_SIZE)
    pc = PointCloud.from_xyz(xyz)
    down = pc.voxel_downsample(
        VOXEL_SIZE, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
    )

    print(
        f"leaf {VOXEL_SIZE} over a +/-500 m extent: PCL returned "
        f"{int(pcl['voxel_out']):,} points (refused={bool(pcl['voxel_refused'])}), "
        f"pcl-rustic returned {len(down):,}"
    )
    assert pcl["voxel_refused"] == 1.0
    assert int(pcl["voxel_out"]) == PCL_POINTS
    assert len(down) < PCL_POINTS


# -- laspy comparisons ---------------------------------------------------


def test_vs_laspy_read(tmp_path: Path) -> None:
    xyz = _gaussian_xyz(IO_POINTS, seed=3)
    rng = np.random.default_rng(4)

    source = PointCloud.from_xyz(xyz)
    source["intensity"] = rng.integers(0, 65535, size=IO_POINTS).astype(np.uint16)
    source["gps_time"] = rng.normal(size=IO_POINTS) * 1.0e3
    path = tmp_path / "bench.las"
    source.write(str(path))

    def theirs_op() -> np.ndarray:
        las = laspy.read(str(path))
        return np.column_stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
        )

    ours, ours_xyz = _timed(lambda: PointCloud.read(str(path)).xyz)
    theirs, theirs_xyz = _timed(theirs_op)

    _row("LAS read + xyz", IO_POINTS, ours, theirs, "laspy ")
    # Coordinates are NOT bit-identical to laspy: ADR-0002 stores an f64
    # offset plus f32 relative coordinates, so reconstruction carries f32
    # epsilon at the cloud's relative extent. The contract is that this stays
    # below the LAS scale quantization (1e-3 by default), not that it is zero.
    las_scale = 1.0e-3
    max_dev = float(np.abs(ours_xyz - theirs_xyz).max())
    print(
        f"{'  coordinate fidelity':<26} max deviation vs laspy = {max_dev:.3e} m "
        f"({max_dev / las_scale:.1%} of the {las_scale} LAS scale step)"
    )
    assert max_dev < las_scale, (
        f"deviation {max_dev} exceeds LAS quantization {las_scale}"
    )


def test_vs_laspy_dtype_parity(tmp_path: Path) -> None:
    """The dtype-promise claim, checked against laspy on a real file."""
    xyz = _gaussian_xyz(1000, seed=5)
    rng = np.random.default_rng(6)

    source = PointCloud.from_xyz(xyz)
    source["intensity"] = rng.integers(0, 65535, size=1000).astype(np.uint16)
    source["classification"] = rng.integers(0, 11, size=1000).astype(np.uint8)
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
