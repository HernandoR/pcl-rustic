#!/usr/bin/env python3
"""
PCL Rustic 测试数据生成工具

生成小规模的LAS/CSV测试数据集
"""

import os
import csv
from pathlib import Path


def generate_test_csv(
    filename: str = "tests/data/test_points.csv", num_points: int = 100
):
    """生成CSV格式的测试点云数据"""

    Path("tests/data").mkdir(parents=True, exist_ok=True)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        # 写入数据（无header）
        for i in range(num_points):
            x = i * 0.1
            y = i * 0.1
            z = i * 0.05
            writer.writerow([f"{x:.1f}", f"{y:.1f}", f"{z:.1f}"])

    print(f"✓ 生成CSV测试数据: {filename} ({num_points}个点)")


def generate_test_las(
    filename: str = "tests/data/test_points.las", num_points: int = 100
):
    """
    生成LAS格式的测试点云数据
    （需要las-rs库）
    """

    try:
        import laspy
    except ImportError:
        print("⚠ laspy未安装，跳过LAS生成")
        print("  安装: pip install laspy")
        return

    Path("tests/data").mkdir(parents=True, exist_ok=True)

    # 创建LAS文件
    las = laspy.create()

    # 生成点云数据
    points_x = [i * 0.1 for i in range(num_points)]
    points_y = [i * 0.1 for i in range(num_points)]
    points_z = [i * 0.05 for i in range(num_points)]

    # 设置坐标
    las.x = points_x
    las.y = points_y
    las.z = points_z

    # 设置强度（如果支持）
    if hasattr(las, "intensity"):
        las.intensity = [i * 100 for i in range(num_points)]

    # 保存
    las.write(filename)
    print(f"✓ 生成LAS测试数据: {filename} ({num_points}个点)")


def generate_all_test_data():
    """生成所有测试数据"""
    print("\n生成PCL Rustic测试数据集")
    print("=" * 50)

    # 小规模测试数据
    generate_test_csv("tests/data/small.csv", 10)
    generate_test_csv("tests/data/medium.csv", 100)

    # 中等规模测试数据
    generate_test_csv("tests/data/large.csv", 1000)

    # LAS测试数据
    generate_test_las("tests/data/test_small.las", 10)
    generate_test_las("tests/data/test_medium.las", 100)

    print("\n✓ 所有测试数据生成完成！")
    print("\n数据文件位置:")
    for f in sorted(Path("tests/data").glob("*")):
        size = os.path.getsize(f) / 1024
        print(f"  - {f.relative_to('tests/data')}: {size:.1f} KB")


if __name__ == "__main__":
    generate_all_test_data()
