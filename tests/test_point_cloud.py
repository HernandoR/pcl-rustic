"""Core PointCloud behavior: construction, geometry, downsampling, dims."""

from __future__ import annotations

import numpy as np
import pytest
from pcl_rustic import DownsampleStrategy, PointCloud


def _rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestConstruction:
    def test_empty_point_cloud(self) -> None:
        pc = PointCloud()
        assert len(pc) == 0
        assert pc.dims == []

    def test_from_xyz_numpy_array(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        assert len(pc) == small_xyz.shape[0]
        np.testing.assert_allclose(pc.xyz, small_xyz, atol=1e-4)

    def test_from_xyz_python_list(self) -> None:
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        pc = PointCloud.from_xyz(data)
        assert len(pc) == 2
        np.testing.assert_allclose(pc.xyz, np.asarray(data), atol=1e-4)

    def test_from_xyz_coerces_integer_dtype(self) -> None:
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        pc = PointCloud.from_xyz(data)
        # Readout dtype follows the cloud's coordinate dtype (float32 by
        # default); the integer input is still ingested via exact f64.
        assert pc.xyz.dtype == pc.dtype
        np.testing.assert_allclose(pc.xyz, data.astype(np.float64))

    def test_from_xyz_rejects_wrong_shape(self) -> None:
        with pytest.raises(TypeError):
            PointCloud.from_xyz(np.zeros((10, 2)))

    def test_from_numpy_with_xyz_key(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_numpy({"xyz": small_xyz})
        assert len(pc) == small_xyz.shape[0]
        np.testing.assert_allclose(pc.xyz, small_xyz, atol=1e-4)

    def test_from_numpy_with_xyz_components(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_numpy(
            {"x": small_xyz[:, 0], "y": small_xyz[:, 1], "z": small_xyz[:, 2]}
        )
        np.testing.assert_allclose(pc.xyz, small_xyz, atol=1e-4)

    def test_from_numpy_sets_standard_extra_dims(self, small_xyz: np.ndarray) -> None:
        intensity = np.arange(small_xyz.shape[0], dtype=np.uint32)
        pc = PointCloud.from_numpy({"xyz": small_xyz, "intensity": intensity})
        assert pc["intensity"].dtype == np.uint16
        np.testing.assert_array_equal(pc["intensity"], intensity.astype(np.uint16))

    def test_from_numpy_infers_extra_dim_dtype(self, small_xyz: np.ndarray) -> None:
        custom = np.arange(small_xyz.shape[0], dtype=np.int32)
        pc = PointCloud.from_numpy({"xyz": small_xyz, "custom": custom})
        assert pc.has_dim("custom")
        assert pc.dtypes["custom"] == np.dtype(np.int32)
        np.testing.assert_array_equal(pc["custom"], custom)

    def test_from_numpy_requires_xyz_or_components(self) -> None:
        with pytest.raises(ValueError):
            PointCloud.from_numpy({"intensity": np.zeros(5, dtype=np.uint16)})


class TestDimensionAccess:
    def test_getitem_setitem(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        pc["classification"] = np.ones(len(pc), dtype=np.uint8)
        np.testing.assert_array_equal(
            pc["classification"], np.ones(len(pc), dtype=np.uint8)
        )

    def test_attribute_sugar_for_standard_dim(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        pc.intensity = np.arange(len(pc), dtype=np.uint16)
        np.testing.assert_array_equal(pc.intensity, np.arange(len(pc), dtype=np.uint16))

    def test_attribute_sugar_for_extra_dim(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        pc.add_extra_dim("elevation", "f4")
        pc.elevation = np.ones(len(pc), dtype=np.float32) * 3.0
        np.testing.assert_array_equal(
            pc.elevation, np.full(len(pc), 3.0, dtype=np.float32)
        )

    def test_attribute_sugar_unknown_name_raises_attribute_error(
        self, small_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        with pytest.raises(AttributeError):
            _ = pc.not_a_real_dimension

    def test_dims_and_dtypes_properties(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        pc["intensity"] = np.zeros(len(pc), dtype=np.uint16)
        pc["classification"] = np.zeros(len(pc), dtype=np.uint8)
        assert set(pc.dims) >= {"x", "y", "z", "intensity", "classification"}
        assert pc.dtypes["intensity"] == np.dtype(np.uint16)
        assert pc.dtypes["classification"] == np.dtype(np.uint8)


class TestCoordinatePrecision:
    def test_utm_scale_precision_under_one_millimeter(
        self, utm_xyz: np.ndarray, float64_dtype: None
    ) -> None:
        """The sub-millimeter UTM promise belongs to the float64 dtype: a
        float32 readout at ~4e6 magnitude cannot beat ~0.25 m by
        construction (f32 has 24 mantissa bits)."""
        pc = PointCloud.from_xyz(utm_xyz)
        out = pc.xyz
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, utm_xyz, atol=1e-3)

    def test_utm_scale_float32_readout_is_ulp_bounded(
        self, utm_xyz: np.ndarray
    ) -> None:
        """Under the float32 default the *readout* quantizes to the f32 grid
        (~0.25 m at 4e6), but must not exceed it: internal storage is f32
        relative to an f64 offset, so no extent-scale error accumulates."""
        pc = PointCloud.from_xyz(utm_xyz)
        out = pc.xyz
        assert out.dtype == np.float32
        max_ulp = np.spacing(np.abs(utm_xyz).max(axis=0).astype(np.float32))
        np.testing.assert_allclose(
            out.astype(np.float64), utm_xyz, atol=float(max_ulp.max())
        )


class TestTransform:
    def test_transform_4x4_matches_numpy_reference(
        self, medium_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        rotation = _rotation_matrix(0.3)
        translation = np.array([1.0, -2.0, 0.5])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        out = pc.transform(matrix)
        expected = medium_xyz @ rotation.T + translation
        np.testing.assert_allclose(out.xyz, expected, atol=1e-3)

    def test_transform_3x3_matches_numpy_reference(
        self, medium_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        scale = np.diag([2.0, 0.5, 1.0])
        out = pc.transform(scale)
        expected = medium_xyz @ scale.T
        np.testing.assert_allclose(out.xyz, expected, atol=1e-3)

    def test_rigid_transform_matches_numpy_reference(
        self, medium_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        rotation = _rotation_matrix(-0.7)
        translation = np.array([5.0, 5.0, -5.0])
        out = pc.rigid_transform(rotation, translation)
        expected = medium_xyz @ rotation.T + translation
        np.testing.assert_allclose(out.xyz, expected, atol=1e-3)

    def test_transform_returns_new_cloud_and_does_not_mutate_source(
        self, small_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        out = pc.transform(np.eye(4))
        assert out is not pc
        np.testing.assert_allclose(pc.xyz, small_xyz, atol=1e-4)

    def test_transform_inplace_matches_copying_path(
        self, medium_xyz: np.ndarray
    ) -> None:
        """inplace=True must be a pure optimization: same values as the
        copying path bit for bit, returning the mutated cloud itself, with
        attribute dimensions untouched."""
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation_matrix(0.3)
        matrix[:3, 3] = [1.0, -2.0, 0.5]

        pc = PointCloud.from_xyz(medium_xyz)
        pc["intensity"] = (np.arange(len(pc)) % 60000).astype(np.uint16)
        expected = pc.transform(matrix).xyz

        out = pc.transform(matrix, inplace=True)
        assert out is pc
        np.testing.assert_array_equal(pc.xyz, expected)
        assert pc["intensity"][1] == 1

    def test_transform_inplace_on_every_available_device(
        self, medium_xyz: np.ndarray
    ) -> None:
        import pcl_rustic

        matrix = np.eye(4)
        matrix[:3, :3] = _rotation_matrix(-0.7)
        matrix[:3, 3] = [5.0, 5.0, -5.0]
        for device in pcl_rustic.available_devices():
            pc = PointCloud.from_xyz(medium_xyz).to_device(device)
            expected = pc.transform(matrix).xyz
            pc.transform(matrix, inplace=True)
            np.testing.assert_array_equal(pc.xyz, expected, err_msg=device)

    def test_rigid_transform_inplace_matches_copying_path(
        self, medium_xyz: np.ndarray
    ) -> None:
        rotation = _rotation_matrix(1.1)
        translation = np.array([0.5, -0.5, 2.0])
        pc = PointCloud.from_xyz(medium_xyz)
        expected = pc.rigid_transform(rotation, translation).xyz
        out = pc.rigid_transform(rotation, translation, inplace=True)
        assert out is pc
        np.testing.assert_array_equal(pc.xyz, expected)

    def test_transform_rejects_bad_matrix_shape(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        with pytest.raises(TypeError):
            pc.transform(np.eye(2))


class TestVoxelDownsample:
    def test_reduces_point_count(self, medium_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        down = pc.voxel_downsample(0.5)
        assert 0 < len(down) <= len(pc)

    def test_random_strategy_is_deterministic_with_seed(
        self, medium_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        a = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.RANDOM, seed=42)
        b = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.RANDOM, seed=42)
        np.testing.assert_array_equal(a.xyz, b.xyz)

    def test_random_strategy_seed_changes_selection(
        self, medium_xyz: np.ndarray
    ) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        a = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.RANDOM, seed=1)
        b = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.RANDOM, seed=2)
        assert len(a) == len(b)
        assert not np.array_equal(a.xyz, b.xyz)

    def test_nearest_to_centroid_is_deterministic(self, medium_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        a = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.NEAREST_TO_CENTROID)
        b = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.NEAREST_TO_CENTROID)
        np.testing.assert_array_equal(a.xyz, b.xyz)

    def test_rejects_non_positive_voxel_size(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        with pytest.raises(ValueError):
            pc.voxel_downsample(-1.0)


class TestCloneLenRepr:
    def test_clone_is_independent(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        pc["intensity"] = np.zeros(len(pc), dtype=np.uint16)
        clone = pc.clone()
        clone["intensity"] = np.ones(len(pc), dtype=np.uint16)
        np.testing.assert_array_equal(
            pc["intensity"], np.zeros(len(pc), dtype=np.uint16)
        )
        np.testing.assert_array_equal(
            clone["intensity"], np.ones(len(pc), dtype=np.uint16)
        )

    def test_len_matches_point_count(self, medium_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(medium_xyz)
        assert len(pc) == medium_xyz.shape[0]

    def test_repr_reports_point_count(self, small_xyz: np.ndarray) -> None:
        pc = PointCloud.from_xyz(small_xyz)
        assert str(len(pc)) in repr(pc)
