"""Throughput benchmarks. Run explicitly with `just bench` (`-m bench`).

These print a small plain-text table; they are not correctness tests beyond
sanity assertions (output size bounds).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from pcl_rustic import DownsampleStrategy, PointCloud

pytestmark = [pytest.mark.bench, pytest.mark.slow]

POINT_COUNTS = [1_000_000, 10_000_000]


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

    t0 = time.perf_counter()
    out = pc.transform(matrix)
    # burn tensor ops are lazy: reading coordinates back forces execution,
    # so materialization must sit inside the timed region.
    result = out.xyz
    elapsed = time.perf_counter() - t0

    throughput = n_points / elapsed if elapsed > 0 else float("inf")
    print(
        f"transform         n={n_points:>11,}  time={elapsed:8.3f}s  "
        f"throughput={throughput:14,.0f} pts/s"
    )
    assert len(out) == n_points
    assert result.shape == (n_points, 3)
