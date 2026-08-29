"""Throughput benchmarks. Run explicitly with `just bench` (`-m bench`).

These print a small plain-text table; they are not correctness tests beyond
sanity assertions (output size bounds).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import resolve_bench_device
from pcl_rustic import DownsampleStrategy, PointCloud, get_default_dtype

pytestmark = [pytest.mark.bench, pytest.mark.slow]

POINT_COUNTS = [1_000_000, 10_000_000]

BENCH_DEVICE = resolve_bench_device()


@pytest.fixture(scope="module", autouse=True)
def _device_banner() -> None:
    """Name the device and dtype up front: every number below depends on
    both, and a run that silently landed on `cpu` is otherwise
    indistinguishable from one that used the GPU."""
    print(f"\n[pcl-rustic device: {BENCH_DEVICE}  dtype: {get_default_dtype()}]")


def _pc(xyz: np.ndarray) -> PointCloud:
    return PointCloud.from_xyz(xyz).to_device(BENCH_DEVICE)


def _gaussian_xyz(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=100.0, size=(n_points, 3))


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_voxel_downsample_throughput(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    pc = _pc(xyz)

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
    pc = _pc(xyz)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    # Warm up first: the first op in a process pays one-time kernel init and
    # (on wgpu) shader compilation, which used to land entirely on the 1M row
    # and made it look slower than the 10M one.
    _ = pc.transform(matrix).xyz

    # Three timing points, because they measure different things:
    #   submit   -- enqueue only (GPU ops are asynchronous),
    #   compute  -- + synchronize(): kernel execution, no data movement,
    #   total    -- + .xyz: readback and numpy materialization.
    # Per-op throughput is quoted on *compute*: materialization is a
    # boundary cost paid once per host readout, not per op, and folding it
    # in made the GPU look slower than the CPU while its kernel was ~100x
    # faster.
    t0 = time.perf_counter()
    out = pc.transform(matrix)
    submit = time.perf_counter() - t0
    out.synchronize()
    compute = time.perf_counter() - t0
    result = out.xyz
    total = time.perf_counter() - t0

    throughput = n_points / compute if compute > 0 else float("inf")
    print(
        f"transform         n={n_points:>11,}  compute={compute * 1e3:8.1f}ms  "
        f"throughput={throughput:14,.0f} pts/s  "
        f"(submit={submit * 1e3:7.1f}ms  materialize={(total - compute) * 1e3:7.1f}ms)"
    )
    assert len(out) == n_points
    assert result.shape == (n_points, 3)


def test_readout_view_vs_copy_throughput(float64_dtype: None) -> None:
    """Zero-copy `view()` vs copying `.xyz`, f64 CPU cloud at 10M points.

    The view should be O(1) -- it wraps the live buffer -- while `.xyz`
    clones 240 MB. This is the counterpart to Open3D's zero-copy `.numpy()`
    the fairness notes call out; it exists only for f64/CPU because the f32
    representations are offset-relative and must compute at readout.
    """
    n = POINT_COUNTS[-1]
    pc = PointCloud.from_xyz(_gaussian_xyz(n))
    assert pc.device == "cpu" and pc.dtype == np.float64

    t0 = time.perf_counter()
    copied = pc.xyz
    t_copy = time.perf_counter() - t0
    t0 = time.perf_counter()
    viewed = pc.view()
    t_view = time.perf_counter() - t0

    print(
        f"readout           n={n:>11,}  .xyz(copy)={t_copy * 1e3:8.1f}ms  "
        f"view()={t_view * 1e6:8.1f}us  ({t_copy / max(t_view, 1e-9):,.0f}x)"
    )
    assert np.shares_memory(viewed, pc.view())
    assert not np.shares_memory(viewed, copied)
    np.testing.assert_array_equal(viewed, copied)
