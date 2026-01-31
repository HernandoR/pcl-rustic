# 下采样

点云下采样功能，提供体素下采样和多种降采样策略。

## API 列表

### DownsampleStrategy

- `DownsampleStrategy.RANDOM`: 随机采样
- `DownsampleStrategy.CENTROID`: 质心采样

### 体素下采样

- `PointCloud.voxel_downsample(voxel_size: float, strategy: int = DownsampleStrategy.CENTROID) -> PointCloud`

## 降采样策略

### CENTROID (质心)

选择体素内所有点的几何质心，适合保持几何形状。

### RANDOM (随机)

随机选择体素内的一个点，速度最快。

## 使用示例

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

xyz = np.random.randn(1_000_000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.CENTROID
)

print(f"原始: {pc.point_count():,} 点")
print(f"下采样: {pc_down.point_count():,} 点")
```

## 相关链接

- [PointCloud](pointcloud.md) - 点云核心类
- [性能基准](../performance/benchmarks.md) - 详细性能数据    (DownsampleStrategy.INTENSITY_CENTROID, "强度质心"),
]

for strategy, name in strategies:
    start = time.time()
    pc_down = pc.voxel_downsample(0.15, strategy)
    elapsed = time.time() - start

    print(f"{name}:")
    print(f"  输出点数: {pc_down.point_count():,}")
    print(f"  耗时: {elapsed:.2f}s")
```

## 性能基准

基准测试（MacBook M1，10M 点云）：

| 体素大小 | 策略 | 输出点数 | 耗时 | 吞吐量 |
|---------|------|---------|------|--------|
| 0.15m | RANDOM | 7.9M | 5.82s | 1.7M/s |
| 0.15m | CENTROID | 7.9M | 7.13s | 1.4M/s |
| 0.15m | INTENSITY_CENTROID | 7.9M | 8.45s | 1.2M/s |

查看 [性能基准测试](../performance/benchmarks.md) 了解更多详情。

## 最佳实践

### 选择体素大小

!!! tip "体素大小选择指南"
    - **0.05-0.10m**: 高精度应用（配准、重建）
    - **0.10-0.20m**: 通用处理
    - **0.20-0.50m**: 快速预览、可视化
    - **>0.50m**: 粗略分析

### 选择降采样策略

| 应用场景 | 推荐策略 | 原因 |
|---------|---------|------|
| 点云配准 | CENTROID | 保持几何精度 |
| 特征提取 | INTENSITY_CENTROID | 保留高强度特征 |
| 快速可视化 | RANDOM | 速度最快 |
| 地面检测 | CENTROID | 平衡性能和精度 |
| 物体识别 | INTENSITY_CENTROID | 保留关键特征 |

### 性能优化

```python
# ✅ 好：选择合适的体素大小
voxel_size = 0.15  # 根据点云密度调整

# ❌ 差：体素过小导致输出点数过多
voxel_size = 0.001  # 几乎没有减少

# ✅ 好：及时释放内存
pc_down = pc.voxel_downsample(0.15)
del pc  # 释放原始点云

# ❌ 差：保留多个副本
pc1 = pc.voxel_downsample(0.10)
pc2 = pc.voxel_downsample(0.15)
pc3 = pc.voxel_downsample(0.20)
```

## 相关链接

- [PointCloud](pointcloud.md) - 点云核心类
- [性能基准](../performance/benchmarks.md) - 详细性能数据
- [优化指南](../performance/optimization.md) - 性能优化技巧
