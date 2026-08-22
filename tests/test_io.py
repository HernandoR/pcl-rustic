"""Multi-format file I/O round-trips (ADR-0004): LAS/LAZ, CSV, Parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pcl_rustic import PointCloud


def _sample_cloud(n: int = 200, seed: int = 0) -> tuple[PointCloud, np.ndarray]:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)) * 100.0
    pc = PointCloud.from_xyz(xyz)
    pc["intensity"] = rng.integers(0, 65535, size=n).astype(np.uint16)
    classification = rng.integers(0, 31, size=n).astype(np.uint8)
    # Code 12 is reserved for the legacy overlap-point encoding; the las
    # crate rejects it on write (see src/io/las.rs module comment).
    classification[classification == 12] = 11
    pc["classification"] = classification
    pc["gps_time"] = rng.normal(size=n).astype(np.float64) * 1.0e6
    pc["red"] = rng.integers(0, 65535, size=n).astype(np.uint16)
    pc["green"] = rng.integers(0, 65535, size=n).astype(np.uint16)
    pc["blue"] = rng.integers(0, 65535, size=n).astype(np.uint16)
    return pc, xyz


class TestParquetRoundTrip:
    def test_lossless_round_trip_with_extra_dims(self, tmp_path: Path) -> None:
        pc, _ = _sample_cloud()
        pc.add_extra_dim("custom_i4", "i4")
        pc["custom_i4"] = np.arange(len(pc), dtype=np.int32)
        pc.add_extra_dim("custom_f4", "f4")
        pc["custom_f4"] = np.linspace(0.0, 1.0, len(pc), dtype=np.float32)

        path = tmp_path / "cloud.parquet"
        pc.write(str(path))
        loaded = PointCloud.read(str(path))

        assert len(loaded) == len(pc)
        np.testing.assert_allclose(loaded.xyz, pc.xyz, atol=1e-9)
        for name in (
            "intensity",
            "classification",
            "gps_time",
            "red",
            "green",
            "blue",
            "custom_i4",
            "custom_f4",
        ):
            assert loaded.dtypes[name] == pc.dtypes[name]
            np.testing.assert_array_equal(loaded[name], pc[name])

    def test_write_read_with_column_mapping(self, tmp_path: Path) -> None:
        pc, _ = _sample_cloud(n=20)
        path = tmp_path / "mapped.parquet"
        columns = {"x": "pos_x", "y": "pos_y", "z": "pos_z"}
        pc.write(str(path), columns=columns)
        loaded = PointCloud.read(str(path), columns=columns)
        np.testing.assert_allclose(loaded.xyz, pc.xyz, atol=1e-9)


class TestCsvRoundTrip:
    def test_round_trip_standard_dims(self, tmp_path: Path) -> None:
        pc, _ = _sample_cloud(n=50)
        path = tmp_path / "cloud.csv"
        pc.write(str(path))
        loaded = PointCloud.read(str(path))

        assert len(loaded) == len(pc)
        np.testing.assert_allclose(loaded.xyz, pc.xyz, atol=1e-6)
        np.testing.assert_array_equal(loaded["intensity"], pc["intensity"])
        np.testing.assert_array_equal(loaded["classification"], pc["classification"])

    def test_round_trip_with_custom_delimiter(self, tmp_path: Path) -> None:
        pc, _ = _sample_cloud(n=20)
        path = tmp_path / "cloud.tsv"
        pc.write(str(path), delimiter="\t")
        loaded = PointCloud.read(str(path), delimiter="\t")
        np.testing.assert_allclose(loaded.xyz, pc.xyz, atol=1e-6)


class TestLasRoundTrip:
    def test_las_round_trip(self, tmp_path: Path) -> None:
        pc, _ = _sample_cloud(n=500)
        path = tmp_path / "cloud.las"
        pc.write(str(path))
        loaded = PointCloud.read(str(path))

        assert len(loaded) == len(pc)
        # LAS quantizes x/y/z to the header scale factor.
        np.testing.assert_allclose(loaded.xyz, pc.xyz, atol=1e-2)
        assert loaded.dtypes["intensity"] == np.dtype(np.uint16)
        assert loaded.dtypes["classification"] == np.dtype(np.uint8)
        np.testing.assert_array_equal(loaded["intensity"], pc["intensity"])
        np.testing.assert_array_equal(loaded["classification"], pc["classification"])
        np.testing.assert_allclose(loaded["gps_time"], pc["gps_time"], atol=1e-6)
        np.testing.assert_array_equal(loaded["red"], pc["red"])
        np.testing.assert_array_equal(loaded["green"], pc["green"])
        np.testing.assert_array_equal(loaded["blue"], pc["blue"])

    def test_laz_round_trip(self, tmp_path: Path) -> None:
        pc, _ = _sample_cloud(n=500)
        path = tmp_path / "cloud.laz"
        pc.write(str(path))
        loaded = PointCloud.read(str(path))

        assert len(loaded) == len(pc)
        np.testing.assert_allclose(loaded.xyz, pc.xyz, atol=1e-2)
        np.testing.assert_array_equal(loaded["intensity"], pc["intensity"])
        np.testing.assert_array_equal(loaded["classification"], pc["classification"])

    def test_module_level_read_alias(self, tmp_path: Path) -> None:
        import pcl_rustic

        pc, _ = _sample_cloud(n=10)
        path = tmp_path / "alias.las"
        pc.write(str(path))
        loaded = pcl_rustic.read(str(path))
        assert len(loaded) == len(pc)
