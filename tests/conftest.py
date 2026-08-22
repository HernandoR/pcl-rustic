"""Shared pytest fixtures for the pcl-rustic test suite."""

from __future__ import annotations

import numpy as np
import pytest


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
