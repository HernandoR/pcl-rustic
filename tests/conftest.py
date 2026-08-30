"""Shared pytest fixtures for the pcl-rustic test suite."""

from __future__ import annotations

import os

import numpy as np
import pcl_rustic as pcl
import pytest


def resolve_bench_device() -> str:
    """The device benchmark modules pin clouds to, from `PCL_BENCH_DEVICE`.

    `cpu` / an explicit device name select that device; `gpu` selects the
    first available non-cpu device and *skips* (rather than silently running
    on cpu) when there is none -- a GPU benchmark that quietly measures the
    CPU is exactly the failure mode this suite already had once. Unset falls
    back to `default_device()`.
    """
    requested = os.environ.get("PCL_BENCH_DEVICE", "")
    if not requested:
        return pcl.default_device()
    if requested == "gpu":
        gpus = [d for d in pcl.available_devices() if d != "cpu"]
        if not gpus:
            # Called at module scope by the bench modules, so the whole
            # module skips rather than erroring at collection.
            pytest.skip(
                "PCL_BENCH_DEVICE=gpu but no GPU device is available",
                allow_module_level=True,
            )
        return gpus[0]
    return requested


@pytest.fixture
def float64_dtype():
    """Pin the default coordinate dtype to float64 for one test, restoring
    the previous value afterwards. For tests that assert exact f64 fidelity
    (laspy parity, lossless round-trips), which the float32 default
    intentionally trades away."""
    previous = pcl.get_default_dtype()
    pcl.set_default_dtype("float64")
    yield
    pcl.set_default_dtype(previous)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for deterministic test data."""
    return np.random.default_rng(0)


@pytest.fixture
def small_xyz(rng: np.random.Generator) -> np.ndarray:
    """10 points, unit scale."""
    return rng.normal(size=(10, 3))


@pytest.fixture
def medium_xyz(rng: np.random.Generator) -> np.ndarray:
    """1,000 points, unit scale."""
    return rng.normal(size=(1_000, 3))


@pytest.fixture
def utm_xyz(rng: np.random.Generator) -> np.ndarray:
    """1,000 points at UTM-like scale (~5e5) to exercise offset storage."""
    center = np.array([5.0e5, 4.0e6, 3.0e2])
    return rng.normal(size=(1_000, 3)) * 10.0 + center
