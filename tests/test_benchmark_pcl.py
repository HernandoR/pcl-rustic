"""Comparison benchmark against PCL (the C++ library).

PCL has no working Python binding on modern Python (`pclpy` pins `==3.6.*`
with a Windows-only wheel; `python-pcl`'s newest release is a 2018 cp27/cp35
alpha), so the baseline is a small C++ harness in `bench/pcl_cpp/`. Build it
with `just bench-pcl`; this module skips when the binary is absent.

Fairness notes:

* All three engines consume byte-identical input. The harness reads the same
  f64 xyz dump this module writes, and Open3D/pcl-rustic are fed the same
  numpy array it was written from.
* Cloud construction is outside every timed region on all sides.
* Element types differ by design: PCL's `PointXYZ` is f32 (padded to 16
  bytes), Open3D is f64, pcl-rustic is f32 relative to an f64 offset. PCL is
  doing the least precise arithmetic here.
* PCL's `VoxelGrid` indexes voxels with a 32-bit integer and silently returns
  the input unfiltered when the grid would overflow it. The harness reports
  that case and this module asserts we noticed, rather than scoring a no-op
  as a fast run.
"""

from __future__ import annotations

import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from pcl_rustic import DownsampleStrategy, PointCloud

pytestmark = [pytest.mark.bench, pytest.mark.slow]

BENCH_BIN = (
    Path(__file__).resolve().parents[1] / "bench" / "pcl_cpp" / "build" / "pcl_bench"
)

# PCL's VoxelGrid overflows its int32 voxel index for a 0.15 leaf over this
# extent, so the comparable run uses a coarser leaf. The refusal itself is
# exercised separately below.
COMPARABLE_LEAF = 1.0
FINE_LEAF = 0.15
N_POINTS = 2_000_000


def _require_binary() -> None:
    if not BENCH_BIN.exists():
        pytest.skip(f"PCL harness not built ({BENCH_BIN}); run `just bench-pcl`")


def _gaussian_xyz(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=100.0, size=(n_points, 3))


def _dump(xyz: np.ndarray, path: Path) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<q", xyz.shape[0]))
        fh.write(np.ascontiguousarray(xyz, dtype="<f8").tobytes())


def _run_harness(path: Path, leaf: float) -> dict[str, float]:
    proc = subprocess.run(
        [str(BENCH_BIN), str(path), str(leaf)],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = dict(tok.split("=", 1) for tok in proc.stdout.split())
    return {k: float(v) for k, v in fields.items()}


def test_compare_pcl_cpp(tmp_path: Path) -> None:
    _require_binary()
    xyz = _gaussian_xyz(N_POINTS)
    data = tmp_path / "cloud.bin"
    _dump(xyz, data)

    pcl_cpp = _run_harness(data, COMPARABLE_LEAF)
    assert not pcl_cpp["voxel_refused"], "PCL refused the leaf size; comparison invalid"

    pc = PointCloud.from_xyz(xyz)

    t0 = time.perf_counter()
    down = pc.voxel_downsample(
        COMPARABLE_LEAF, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
    )
    ours_voxel = time.perf_counter() - t0

    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    t0 = time.perf_counter()
    moved = pc.transform(matrix).xyz
    ours_transform = time.perf_counter() - t0

    def verdict(ours: float, theirs: float) -> str:
        ratio = theirs / ours if ours > 0 else float("inf")
        return f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"

    print(
        f"voxel_downsample (leaf {COMPARABLE_LEAF})  n={N_POINTS:,}  "
        f"pcl-rustic={ours_voxel:7.3f}s  pcl-c++={pcl_cpp['voxel_secs']:7.3f}s  "
        f"-> pcl-rustic is {verdict(ours_voxel, pcl_cpp['voxel_secs'])}  "
        f"(out: {len(down):,} vs {int(pcl_cpp['voxel_out']):,})"
    )
    print(
        f"transform (4x4)                n={N_POINTS:,}  "
        f"pcl-rustic={ours_transform:7.3f}s  pcl-c++={pcl_cpp['transform_secs']:7.3f}s  "
        f"-> pcl-rustic is {verdict(ours_transform, pcl_cpp['transform_secs'])}"
    )
    assert moved.shape == (N_POINTS, 3)


def test_pcl_voxelgrid_refuses_fine_leaf(tmp_path: Path) -> None:
    """PCL silently no-ops where pcl-rustic downsamples; document it."""
    _require_binary()
    xyz = _gaussian_xyz(N_POINTS)
    data = tmp_path / "cloud.bin"
    _dump(xyz, data)

    pcl_cpp = _run_harness(data, FINE_LEAF)
    pc = PointCloud.from_xyz(xyz)
    down = pc.voxel_downsample(
        FINE_LEAF, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
    )

    print(
        f"leaf {FINE_LEAF} over a +/-500 m extent: PCL returned "
        f"{int(pcl_cpp['voxel_out']):,} points (refused={bool(pcl_cpp['voxel_refused'])}), "
        f"pcl-rustic returned {len(down):,}"
    )
    # PCL's int32 voxel index overflows here and it returns the cloud as-is.
    assert pcl_cpp["voxel_refused"] == 1.0
    assert int(pcl_cpp["voxel_out"]) == N_POINTS
    # We actually downsample.
    assert len(down) < N_POINTS
