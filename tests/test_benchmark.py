"""Throughput benchmarks. Run explicitly with `just bench` (`-m bench`).

These print a small plain-text table; they are not correctness tests beyond
sanity assertions (output size bounds).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from pcl_rustic import DownsampleStrategy, PointCloud, default_device

pytestmark = [pytest.mark.bench, pytest.mark.slow]

POINT_COUNTS = [1_000_000, 10_000_000]


@pytest.fixture(scope="module", autouse=True)
def _device_banner() -> None:
    """Name the device up front: every number below is device-dependent, and
    a run that silently landed on `cpu` is otherwise indistinguishable from
    one that used the GPU."""
    print(f"\n[pcl-rustic device: {default_device()}]")


def _gaussian_xyz(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=100.0, size=(n_points, 3))


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_voxel_downsample_throughput(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    pc = PointCloud.from_xyz(xyz)

    t0 = time.perf_counter()
    down = pc.voxel_downsample(0.15, strategy=DownsampleStrategy.NEAREST_TO_CENTROID)
    elapsed = time.perf_counter() - t0

    throughput = n_points / elapsed if elapsed > 0 else float("inf")
    print(
        f"voxel_downsample  n={n_points:>11,}  out={len(down):>11,}  "
        f"time={elapsed:8.3f}s  throughput={throughput:14,.0f} pts/s"
    )
    assert 0 < len(down) <= n_points


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_transform_throughput(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    pc = PointCloud.from_xyz(xyz)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    # Warm up first: the first op in a process pays one-time kernel init and
    # (on wgpu) shader compilation, which used to land entirely on the 1M row
    # and made it look slower than the 10M one.
    _ = pc.transform(matrix).xyz

    t0 = time.perf_counter()
    out = pc.transform(matrix)
    submit = time.perf_counter() - t0
    # burn tensor ops are lazy: reading coordinates back forces execution,
    # so materialization must sit inside the timed region.
    result = out.xyz
    elapsed = time.perf_counter() - t0

    throughput = n_points / elapsed if elapsed > 0 else float("inf")
    # Split the two, because they scale differently and the total is
    # misleading on its own: on a GPU the transform itself is sub-millisecond
    # and the time is readback over PCIe plus the f32 -> f64 widening pass
    # into a fresh (N, 3) f64 array. A GPU that looks "slower" here is
    # usually a materialization cost, not idle silicon.
    print(
        f"transform         n={n_points:>11,}  time={elapsed:8.3f}s  "
        f"throughput={throughput:14,.0f} pts/s  "
        f"(submit={submit * 1e3:7.1f}ms  materialize={(elapsed - submit) * 1e3:7.1f}ms)"
    )
    assert len(out) == n_points
    assert result.shape == (n_points, 3)
