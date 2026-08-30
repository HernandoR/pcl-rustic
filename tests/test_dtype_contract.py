"""Tests for the ADR-0002 LAS-aligned dimension & dtype contract.

Every standard dimension carries a pinned numpy dtype. Setters accept the
exact dtype as-is, coerce same-kind arrays (with range validation), and
reject kind-incompatible input. Extra dimensions round-trip through the
laspy-aligned allowed dtype set.
"""

from __future__ import annotations

import numpy as np
import pytest
from pcl_rustic import STANDARD_DIMENSIONS, PointCloud

_COORDINATE_DIMS = ("x", "y", "z")
_ATTRIBUTE_DIMS = sorted(
    name for name in STANDARD_DIMENSIONS if name not in _COORDINATE_DIMS
)

_EXTRA_DTYPES: dict[str, np.dtype] = {
    "u1": np.dtype(np.uint8),
    "u2": np.dtype(np.uint16),
    "u4": np.dtype(np.uint32),
    "u8": np.dtype(np.uint64),
    "i1": np.dtype(np.int8),
    "i2": np.dtype(np.int16),
    "i4": np.dtype(np.int32),
    "i8": np.dtype(np.int64),
    "f4": np.dtype(np.float32),
    "f8": np.dtype(np.float64),
}


def _sample_value(dtype: np.dtype, n: int) -> np.ndarray:
    """A small array of `n` values that is always in-range for `dtype`."""
    if dtype.kind == "b":
        return np.arange(n) % 2 == 0
    if dtype.kind in ("i", "u"):
        info = np.iinfo(dtype)
        span = min(int(info.max), 100)
        return (np.arange(n, dtype=np.int64) % (span + 1)).astype(dtype)
    if dtype.kind == "f":
        return (np.arange(n, dtype=np.float64) * 0.5 - 1.25).astype(dtype)
    raise AssertionError(f"unexpected dtype kind: {dtype.kind}")


@pytest.fixture
def cloud(small_xyz: np.ndarray) -> PointCloud:
    return PointCloud.from_xyz(small_xyz)


@pytest.mark.parametrize("name", _ATTRIBUTE_DIMS)
def test_standard_dim_exact_dtype_round_trip(cloud: PointCloud, name: str) -> None:
    dtype = STANDARD_DIMENSIONS[name]
    value = _sample_value(dtype, len(cloud))
    cloud[name] = value
    out = cloud[name]
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, value)


def test_xyz_dtype_follows_default_dtype(small_xyz: np.ndarray) -> None:
    """Coordinate reads return the cloud's dtype, fixed at construction from
    the global default (float32 unless overridden); writes are always taken
    as f64 regardless (the setter contract is unchanged)."""
    import pcl_rustic

    assert pcl_rustic.get_default_dtype() == "float32"
    cloud32 = PointCloud.from_xyz(small_xyz)
    assert cloud32.dtype == np.float32
    for arr in (cloud32.x, cloud32.y, cloud32.z, cloud32.xyz):
        assert arr.dtype == np.float32

    pcl_rustic.set_default_dtype(np.float64)
    try:
        cloud64 = PointCloud.from_xyz(small_xyz)
        assert cloud64.dtype == np.float64
        for arr in (cloud64.x, cloud64.y, cloud64.z, cloud64.xyz):
            assert arr.dtype == np.float64
        # Torch-style: the existing cloud keeps the dtype it was born with.
        assert cloud32.dtype == np.float32
        assert cloud32.xyz.dtype == np.float32
    finally:
        pcl_rustic.set_default_dtype(np.float32)


def test_float64_dtype_clouds_ingest_on_cpu(
    small_xyz: np.ndarray, float64_dtype: None
) -> None:
    """No accelerator representation can hold f64, so an f64 cloud placed on
    a GPU at construction would round immediately -- defeating the opt-in.
    Rounding under f64 dtype must only happen via an explicit to_device."""
    pc = PointCloud.from_xyz(small_xyz)
    assert pc.device == "cpu"
    np.testing.assert_array_equal(pc.xyz, small_xyz)


def test_set_default_dtype_rejects_unsupported() -> None:
    import pcl_rustic

    with pytest.raises(ValueError):
        pcl_rustic.set_default_dtype(np.int32)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("float64", "float64"), ("f32", "float32"), ("bogus", "float32")],
)
def test_default_dtype_env_var(env_value: str, expected: str) -> None:
    """`PCL_RUSTIC_DEFAULT_DTYPE` seeds the default; it is read once at first
    use, so this needs a fresh interpreter. Invalid values must fall back to
    float32 with a warning rather than failing the import (ADR-0001 extends
    the no-panic-at-import rule here)."""
    import os
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pcl_rustic; print(pcl_rustic.get_default_dtype())",
        ],
        env={**os.environ, "PCL_RUSTIC_DEFAULT_DTYPE": env_value},
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == expected
    if expected != env_value and env_value not in ("f32", "f64"):
        assert "ignoring invalid PCL_RUSTIC_DEFAULT_DTYPE" in proc.stderr


class TestSameKindCoercion:
    def test_wider_int_into_narrower_int(self, cloud: PointCloud) -> None:
        values = np.arange(len(cloud), dtype=np.int64) % 5
        cloud["classification"] = values
        out = cloud["classification"]
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, values.astype(np.uint8))

    def test_narrower_int_into_wider_int(self, cloud: PointCloud) -> None:
        values = (np.arange(len(cloud), dtype=np.int8) % 5).astype(np.int8)
        cloud["scan_angle"] = values
        out = cloud["scan_angle"]
        assert out.dtype == np.int16
        np.testing.assert_array_equal(out, values.astype(np.int16))

    def test_float32_into_float64_dim(self, cloud: PointCloud) -> None:
        values = np.linspace(0.0, 1.0, len(cloud), dtype=np.float32)
        cloud["gps_time"] = values
        out = cloud["gps_time"]
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, values.astype(np.float64))

    def test_int_into_float_dim(self, cloud: PointCloud) -> None:
        values = np.arange(len(cloud), dtype=np.int32)
        cloud["gps_time"] = values
        assert cloud["gps_time"].dtype == np.float64
        np.testing.assert_array_equal(cloud["gps_time"], values.astype(np.float64))

    def test_unsigned_into_signed_int(self, cloud: PointCloud) -> None:
        values = np.arange(len(cloud), dtype=np.uint8)
        cloud["scan_angle"] = values
        np.testing.assert_array_equal(cloud["scan_angle"], values.astype(np.int16))


class TestOverflowAndKindErrors:
    def test_overflow_raises(self, cloud: PointCloud) -> None:
        values = np.full(len(cloud), 300, dtype=np.int64)
        with pytest.raises(OverflowError):
            cloud["classification"] = values  # u1 max is 255

    def test_overflow_raises_for_negative_into_unsigned(
        self, cloud: PointCloud
    ) -> None:
        values = np.full(len(cloud), -1, dtype=np.int32)
        with pytest.raises(OverflowError):
            cloud["intensity"] = values  # u2 has no negative range

    def test_float_into_int_raises_type_error(self, cloud: PointCloud) -> None:
        values = np.full(len(cloud), 1.5, dtype=np.float64)
        with pytest.raises(TypeError):
            cloud["classification"] = values

    def test_int_into_bool_raises_type_error(self, cloud: PointCloud) -> None:
        values = np.zeros(len(cloud), dtype=np.int32)
        with pytest.raises(TypeError):
            cloud["synthetic"] = values

    def test_bool_dim_round_trip(self, cloud: PointCloud) -> None:
        values = np.arange(len(cloud)) % 2 == 0
        cloud["synthetic"] = values
        out = cloud["synthetic"]
        assert out.dtype == np.bool_
        np.testing.assert_array_equal(out, values)


@pytest.mark.parametrize("code", sorted(_EXTRA_DTYPES))
def test_extra_dim_round_trip_for_every_allowed_dtype(
    cloud: PointCloud, code: str
) -> None:
    dtype = _EXTRA_DTYPES[code]
    name = f"extra_{code}"
    cloud.add_extra_dim(name, code)
    assert cloud.has_dim(name)
    assert cloud.dtypes[name] == dtype

    value = _sample_value(dtype, len(cloud))
    cloud[name] = value
    out = cloud[name]
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, value)


def test_add_extra_dim_accepts_numpy_dtype_name(cloud: PointCloud) -> None:
    cloud.add_extra_dim("elevation", "float32")
    assert cloud.has_dim("elevation")
    assert cloud.dtypes["elevation"] == np.dtype(np.float32)


def test_extra_dim_is_zero_filled(cloud: PointCloud) -> None:
    cloud.add_extra_dim("flag", "u1")
    np.testing.assert_array_equal(cloud["flag"], np.zeros(len(cloud), dtype=np.uint8))


def test_set_unknown_dim_without_declaring_raises_key_error(cloud: PointCloud) -> None:
    with pytest.raises(KeyError):
        cloud["undeclared"] = np.zeros(len(cloud), dtype=np.uint8)


def test_remove_dim(cloud: PointCloud) -> None:
    cloud["intensity"] = np.zeros(len(cloud), dtype=np.uint16)
    assert cloud.has_dim("intensity")
    cloud.remove_dim("intensity")
    assert not cloud.has_dim("intensity")
