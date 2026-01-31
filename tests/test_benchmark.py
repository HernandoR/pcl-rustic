"""
PCL Rustic 性能基准测试

使用 Gaussian 分布生成大规模点云，测试体素下采样性能
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import pytest
import numpy as np
import time
from loguru import logger
from pcl_rustic import PointCloud, DownsampleStrategy


# 配置 loguru
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)


def generate_gaussian_point_cloud(
    num_points: int,
    x_range: Tuple[float, float] = (-100, 250),
    y_range: Tuple[float, float] = (-100, 250),
    z_range: Tuple[float, float] = (-3, 7),
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    生成高斯分布的点云数据

    Args:
        num_points: 点数
        x_range: X 坐标范围
        y_range: Y 坐标范围
        z_range: Z 坐标范围
        seed: 随机种子

    Returns:
        包含 xyz, intensity, d1, d2 的字典
    """
    if seed is not None:
        np.random.seed(seed)

    # 随机中心点
    center_x = np.random.uniform(x_range[0] + 50, x_range[1] - 50)
    center_y = np.random.uniform(y_range[0] + 50, y_range[1] - 50)
    center_z = np.random.uniform(z_range[0] + 1, z_range[1] - 1)

    # 随机 sigma（标准差）
    sigma_xy = np.random.uniform(30, 80)  # XY 平面扩展
    sigma_z = np.random.uniform(1, 3)  # Z 方向较小

    logger.info(
        f"生成点云: N={num_points:,}, "
        f"center=({center_x:.1f}, {center_y:.1f}, {center_z:.1f}), "
        f"sigma_xy={sigma_xy:.1f}, sigma_z={sigma_z:.1f}"
    )

    # 生成高斯分布坐标
    x = np.random.normal(center_x, sigma_xy, num_points).astype(np.float32)
    y = np.random.normal(center_y, sigma_xy, num_points).astype(np.float32)
    z = np.random.normal(center_z, sigma_z, num_points).astype(np.float32)

    # 裁剪到范围内
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])
    z = np.clip(z, z_range[0], z_range[1])

    # 合并为 [N, 3] 数组
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    # 生成 intensity（高斯分布，中心附近更高）
    dist_from_center = np.sqrt(
        (x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2
    )
    intensity = (1.0 / (1.0 + dist_from_center / sigma_xy) * 255).astype(np.float32)

    # 生成自定义属性 d1, d2（随机高斯分布）
    d1 = np.random.normal(0, 1, num_points).astype(np.float32)
    d2 = np.random.normal(0, 1, num_points).astype(np.float32)

    return {
        "xyz": xyz,
        "intensity": intensity,
        "d1": d1,
        "d2": d2,
    }


class TestBenchmarkVoxelDownsample:
    """体素下采样性能基准测试"""

    @pytest.mark.parametrize(
        "num_points",
        [
            pytest.param(10_000_000, id="10M"),
            pytest.param(50_000_000, id="50M"),
            pytest.param(100_000_000, id="100M", marks=pytest.mark.slow),
        ],
    )
    @pytest.mark.parametrize(
        "voxel_size",
        [
            pytest.param(0.06, id="voxel_0.06"),
            pytest.param(0.15, id="voxel_0.15"),
            pytest.param(0.20, id="voxel_0.20"),
        ],
    )
    def test_voxel_downsample_performance(self, num_points: int, voxel_size: float):
        """测试体素下采样性能"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始测试: {num_points:,} 点, voxel_size={voxel_size}")
        logger.info(f"{'=' * 60}")

        # 生成点云数据
        t0 = time.perf_counter()
        data = generate_gaussian_point_cloud(num_points)
        t_gen = time.perf_counter() - t0
        logger.info(f"数据生成耗时: {t_gen:.2f}s")

        # 创建点云
        t0 = time.perf_counter()
        pc = PointCloud.from_xyz(data["xyz"])
        pc.set_intensity(data["intensity"])
        pc.add_attribute("d1", data["d1"])
        pc.add_attribute("d2", data["d2"])
        t_create = time.perf_counter() - t0
        logger.info(f"点云创建耗时: {t_create:.2f}s")
        logger.info(f"输入点数: {pc.point_count():,}")
        logger.info(f"内存占用: {pc.memory_usage() / 1024 / 1024:.1f} MB")

        # 体素下采样
        t0 = time.perf_counter()
        downsampled = pc.voxel_downsample(voxel_size, DownsampleStrategy.CENTROID)
        t_downsample = time.perf_counter() - t0

        out_count = downsampled.point_count()
        reduction_ratio = (1 - out_count / num_points) * 100

        logger.success(
            f"下采样完成! "
            f"voxel={voxel_size}, "
            f"输入={num_points:,}, "
            f"输出={out_count:,}, "
            f"减少={reduction_ratio:.1f}%, "
            f"耗时={t_downsample:.3f}s"
        )

        # 验证结果
        assert downsampled.point_count() > 0
        assert downsampled.point_count() <= num_points
        assert downsampled.has_intensity()
        assert "d1" in downsampled.attribute_names()
        assert "d2" in downsampled.attribute_names()


class TestBenchmarkSummary:
    """生成完整的性能报告"""

    def test_full_benchmark_report(self):
        """运行完整的性能基准测试并输出报告"""
        logger.info("\n" + "=" * 70)
        logger.info("PCL Rustic 体素下采样性能基准测试")
        logger.info("=" * 70)

        # 测试配置
        point_counts = [10_000_000, 50_000_000]
        voxel_sizes = [0.06, 0.15, 0.20]

        results = []

        for num_points in point_counts:
            # 生成一次数据，测试多种 voxel size
            data = generate_gaussian_point_cloud(num_points, seed=42)

            # 创建点云
            pc = PointCloud.from_xyz(data["xyz"])
            pc.set_intensity(data["intensity"])
            pc.add_attribute("d1", data["d1"])
            pc.add_attribute("d2", data["d2"])

            input_count = pc.point_count()
            memory_mb = pc.memory_usage() / 1024 / 1024

            logger.info(f"\n点云规模: {input_count:,} 点, 内存: {memory_mb:.1f} MB")
            logger.info("-" * 50)

            for voxel_size in voxel_sizes:
                # 下采样
                t0 = time.perf_counter()
                downsampled = pc.voxel_downsample(
                    voxel_size, DownsampleStrategy.CENTROID
                )
                elapsed = time.perf_counter() - t0

                out_count = downsampled.point_count()
                reduction = (1 - out_count / input_count) * 100
                throughput = input_count / elapsed / 1_000_000  # M points/sec

                results.append(
                    {
                        "input": input_count,
                        "voxel_size": voxel_size,
                        "output": out_count,
                        "reduction": reduction,
                        "time": elapsed,
                        "throughput": throughput,
                    }
                )

                logger.info(
                    f"  voxel={voxel_size:.2f}: "
                    f"{input_count:>12,} → {out_count:>10,} "
                    f"({reduction:5.1f}% 减少) "
                    f"| {elapsed:6.3f}s "
                    f"| {throughput:.1f}M pts/s"
                )

        # 输出汇总表
        logger.info("\n" + "=" * 70)
        logger.info("性能汇总")
        logger.info("=" * 70)
        logger.info(
            f"{'输入点数':>12} | {'Voxel':>6} | {'输出点数':>12} | "
            f"{'减少率':>8} | {'耗时':>8} | {'吞吐量':>10}"
        )
        logger.info("-" * 70)

        for r in results:
            logger.info(
                f"{r['input']:>12,} | {r['voxel_size']:>6.2f} | {r['output']:>12,} | "
                f"{r['reduction']:>7.1f}% | {r['time']:>7.3f}s | "
                f"{r['throughput']:>8.1f}M/s"
            )

        logger.info("=" * 70)

        # 断言所有测试通过
        assert len(results) == len(point_counts) * len(voxel_sizes)
        for r in results:
            assert r["output"] > 0
            assert r["reduction"] >= 0


if __name__ == "__main__":
    # 直接运行时执行基准测试
    pytest.main([__file__, "-v", "-s", "-k", "test_full_benchmark_report"])
