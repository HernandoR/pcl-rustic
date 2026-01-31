"""
pcl_rustic 点云库的 pytest 测试用例

覆盖核心功能、边界场景、异常场景
"""

import pytest
import numpy as np
from loguru import logger
from pcl_rustic import PointCloud, DownsampleStrategy
import math


class TestPointCloudLifecycle:
    """点云生命周期测试"""

    def test_create_empty_point_cloud(self):
        """测试创建空点云"""
        pc = PointCloud()
        assert pc.point_count() == 0
        assert not pc.has_intensity()
        assert not pc.has_rgb()

    def test_create_from_xyz(self):
        """测试从 numpy XYZ 数组创建点云"""
        xyz = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
        )
        pc = PointCloud.from_xyz(xyz)
        assert pc.point_count() == 3
        result = pc.get_xyz()
        np.testing.assert_array_almost_equal(result, xyz)

    def test_clone_point_cloud(self):
        """测试点云克隆"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc1 = PointCloud.from_xyz(xyz)
        pc2 = pc1.clone()
        assert pc2.point_count() == 1
        np.testing.assert_array_almost_equal(pc2.get_xyz(), pc1.get_xyz())


class TestPointCloudProperties:
    """点云属性测试"""

    def test_set_intensity(self):
        """测试设置 intensity"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        intensity = np.array([100.0, 200.0], dtype=np.float32)
        pc.set_intensity(intensity)

        assert pc.has_intensity()
        np.testing.assert_array_almost_equal(pc.get_intensity(), intensity)

    def test_set_rgb(self):
        """测试设置 RGB"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        r = np.array([255.0, 0.0], dtype=np.float32)
        g = np.array([0.0, 255.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        pc.set_rgb(r, g, b)

        assert pc.has_rgb()
        result = pc.get_rgb()
        assert result is not None
        np.testing.assert_array_almost_equal(result[0], [255, 0])  # R 通道

    def test_add_custom_attribute(self):
        """测试添加自定义属性"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        attr_data = np.array([1.5, 2.5], dtype=np.float32)
        pc.add_attribute("confidence", attr_data)

        assert "confidence" in pc.attribute_names()
        np.testing.assert_array_almost_equal(pc.get_attribute("confidence"), attr_data)

    def test_add_duplicate_attribute_fails(self):
        """测试添加重复属性时失败"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        pc.add_attribute("test", np.array([1.0], dtype=np.float32))

        with pytest.raises(ValueError):
            pc.add_attribute("test", np.array([2.0], dtype=np.float32))

    def test_set_attribute_overwrites(self):
        """测试设置属性会覆盖"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        pc.add_attribute("test", np.array([1.0], dtype=np.float32))
        pc.set_attribute("test", np.array([2.0], dtype=np.float32))

        np.testing.assert_array_almost_equal(pc.get_attribute("test"), [2.0])

    def test_remove_attribute(self):
        """测试移除属性"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        pc.add_attribute("test", np.array([1.0], dtype=np.float32))
        pc.remove_attribute("test")

        assert "test" not in pc.attribute_names()

    def test_attribute_dimension_mismatch(self):
        """测试属性维度不匹配时失败"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        with pytest.raises(ValueError):
            pc.add_attribute(
                "test", np.array([1.0], dtype=np.float32)
            )  # 只有1个值，点云有2个点


class TestCoordinateTransform:
    """坐标变换测试"""

    def test_3x3_transform_scaling(self):
        """测试 3x3 变换矩阵（缩放）"""
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 2 倍缩放矩阵
        matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

        pc_scaled = pc.transform(matrix)
        result = pc_scaled.get_xyz()

        assert abs(result[0][0] - 2.0) < 1e-5
        assert abs(result[1][1] - 2.0) < 1e-5

    def test_3x3_transform_rotation(self):
        """测试 3x3 变换矩阵（旋转）"""
        xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 90 度绕 Z 轴旋转
        angle = math.pi / 2
        matrix = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]

        pc_rotated = pc.transform(matrix)
        result = pc_rotated.get_xyz()

        # 应接近 [0, 1, 0]
        assert abs(result[0][0]) < 1e-5
        assert abs(result[0][1] - 1.0) < 1e-5

    def test_rigid_transform(self):
        """测试刚体变换（旋转+平移）"""
        xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 恒等旋转
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        # 平移 [1, 2, 3]
        translation = [1.0, 2.0, 3.0]

        pc_transformed = pc.rigid_transform(rotation, translation)
        result = pc_transformed.get_xyz()

        assert abs(result[0][0] - 2.0) < 1e-5
        assert abs(result[0][1] - 2.0) < 1e-5
        assert abs(result[0][2] - 3.0) < 1e-5

    def test_invalid_matrix_dimension(self):
        """测试非法矩阵维度"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 2x2 矩阵（不支持）
        with pytest.raises(ValueError):
            pc.transform([[1.0, 0.0], [0.0, 1.0]])


class TestVoxelDownsample:
    """体素下采样测试"""

    def test_random_downsample(self):
        """测试随机采样下采样"""
        # 创建两个体素中各 2 个点
        xyz = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],  # 体素1
                [1.1, 1.1, 1.1],
                [1.2, 1.2, 1.2],  # 体素2
            ],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)

        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM)

        # 下采样后应该有 2 个点（每个体素 1 个）
        assert downsampled.point_count() <= pc.point_count()
        assert downsampled.point_count() > 0

    def test_centroid_downsample(self):
        """测试重心采样下采样"""
        xyz = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1], [1.2, 1.2, 1.2]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)

        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.CENTROID)

        assert downsampled.point_count() <= pc.point_count()

    def test_downsample_with_intensity(self):
        """测试下采样保留 intensity"""
        xyz = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1]], dtype=np.float32
        )
        intensity = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        pc = PointCloud.from_xyz(xyz)
        pc.set_intensity(intensity)

        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM)

        assert downsampled.has_intensity()
        result_intensity = downsampled.get_intensity()
        assert result_intensity is not None
        assert len(result_intensity) == downsampled.point_count()

    def test_invalid_voxel_size(self):
        """测试无效 voxel_size"""
        xyz = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        with pytest.raises(ValueError):
            pc.voxel_downsample(-1.0, DownsampleStrategy.RANDOM)


class TestMemoryAndRepr:
    """内存和表示测试"""

    def test_memory_usage(self):
        """测试内存占用计算"""
        xyz = np.random.randn(100, 3).astype(np.float32)
        pc = PointCloud.from_xyz(xyz)

        memory = pc.memory_usage()
        assert memory > 0
        # 100 个 3D 点的 f32 数据: 100 * 3 * 4 bytes = 1200 bytes
        assert memory >= 1200

    def test_repr(self):
        """测试点云表示"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        repr_str = repr(pc)
        assert "PointCloud" in repr_str
        assert "1" in repr_str  # 点数


class TestEdgeCases:
    """边界场景测试"""

    def test_single_point(self):
        """测试单点云"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        assert pc.point_count() == 1
        pc.set_intensity(np.array([100.0], dtype=np.float32))
        assert pc.has_intensity()

    def test_large_point_count(self):
        """测试大规模点云（可选性能测试）"""
        n = 10000
        xyz = np.random.randn(n, 3).astype(np.float32)
        pc = PointCloud.from_xyz(xyz)

        assert pc.point_count() == n

    def test_empty_downsample(self):
        """测试空点云下采样"""
        pc = PointCloud()
        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM)
        assert downsampled.point_count() == 0

    def test_zero_points_properties(self):
        """测试空点云的属性操作"""
        pc = PointCloud()

        assert not pc.has_intensity()
        assert not pc.has_rgb()
        assert pc.attribute_names() == []


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 创建点云
        xyz = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1], [1.2, 1.2, 1.2]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)

        # 2. 添加属性
        pc.set_intensity(np.array([100.0, 150.0, 200.0, 250.0], dtype=np.float32))
        pc.set_rgb(
            np.array([255.0, 0.0, 0.0, 255.0], dtype=np.float32),
            np.array([0.0, 255.0, 0.0, 255.0], dtype=np.float32),
            np.array([0.0, 0.0, 255.0, 0.0], dtype=np.float32),
        )
        pc.add_attribute("confidence", np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32))

        # 3. 变换
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        translation = [1.0, 2.0, 3.0]
        pc_transformed = pc.rigid_transform(rotation, translation)

        # 4. 下采样
        pc_downsampled = pc_transformed.voxel_downsample(
            1.0, DownsampleStrategy.CENTROID
        )

        # 5. 验证结果
        assert pc_downsampled.point_count() > 0
        assert pc_downsampled.has_intensity()
        assert pc_downsampled.has_rgb()
        assert "confidence" in pc_downsampled.attribute_names()

    def test_chain_operations(self):
        """测试链式操作"""
        xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
        pc = PointCloud.from_xyz(xyz)

        # 变换 -> 下采样
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = pc.transform(matrix).voxel_downsample(2.0, DownsampleStrategy.RANDOM)

        assert result.point_count() > 0
        assert result.point_count() <= pc.point_count()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
