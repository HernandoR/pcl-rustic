# 下采样

点云下采样功能，提供体素下采样和 2 种降采样策略。

## API 列表

### DownsampleStrategy

| 常量 | 值 | 说明 |
|------|-----|------|
| `DownsampleStrategy.RANDOM` | `0` | 随机采样 — 选择体素中间位置的索引 |
| `DownsampleStrategy.CENTROID` | `1` | 质心采样 — 选择最接近体素几何中心的点 |

### 体素下采样

```
PointCloud.voxel_downsample(voxel_size: float, strategy: int) -> PointCloud
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `voxel_size` | `float` | 体素大小，必须 > 0 |
| `strategy` | `int` | 采样策略，使用 `DownsampleStrategy.RANDOM` 或 `DownsampleStrategy.CENTROID` |

## 降采样策略详解

### CENTROID (质心)

计算体素内所有点的几何质心位置，选择最接近质心的原始点作为代表。适合需要保持几何精度的场景。

### RANDOM (随机)

从体素中选择中间位置的点作为代表。速度最快，适合快速预览和可视化。

!!! note "策略对比"
    | 应用场景 | 推荐策略 | 原因 |
    |---------|---------|------|
    | 点云配准 | CENTROID | 保持几何精度 |
    | 快速可视化 | RANDOM | 速度最快 |
    | 通用处理 | CENTROID | 平衡性能和精度 |

## 使用示例

### 基本用法

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

xyz = np.random.randn(1_000_000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

# 质心采样
pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.CENTROID
)

# 随机采样
pc_down_fast = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.RANDOM
)

print(f"原始: {pc.point_count():,} 点")
print(f"质心下采样: {pc_down.point_count():,} 点")
print(f"随机下采样: {pc_down_fast.point_count():,} 点")
```

### 带属性的下采样

```python
# 创建带属性的点云
intensity = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_intensity(intensity)

r = np.random.randint(0, 256, pc.point_count(), dtype=np.float32)
g = np.random.randint(0, 256, pc.point_count(), dtype=np.float32)
b = np.random.randint(0, 256, pc.point_count(), dtype=np.float32)
pc.set_rgb(r, g, b)

# 下采样 — 所有属性自动跟随下采样
pc_down = pc.voxel_downsample(0.15, DownsampleStrategy.CENTROID)

# 属性保持一致
assert pc_down.has_intensity()
assert pc_down.has_rgb()
```

## 性能基准

基准测试（MacBook M1，10M 点云）：

| 体素大小 | 策略 | 输出点数 | 耗时 | 吞吐量 |
|---------|------|---------|------|--------|
| 0.15m | RANDOM | 7.9M | 5.82s | 1.7M/s |
| 0.15m | CENTROID | 7.9M | 7.13s | 1.4M/s |

## 最佳实践

### 选择体素大小

!!! tip "体素大小选择指南"
    - **0.05–0.10m**: 高精度应用（配准、重建）
    - **0.10–0.20m**: 通用处理
    - **0.20–0.50m**: 快速预览、可视化
    - **> 0.50m**: 粗略分析

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
