# 基本使用

本指南介绍 `pcl-rustic` 的核心功能和常见用法。

## 创建点云

### 从 NumPy 数组

最常用的方式是从 NumPy 数组创建点云。推荐使用 `float32` 以获得最佳性能，但也支持 `float64`、`int32`、`int64`（自动转换）：

```python
import numpy as np
from pcl_rustic import PointCloud

# 推荐：直接使用 float32
xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

# 也支持：float64 等其他 dtype（自动转换）
xyz_f64 = np.random.randn(10000, 3)  # 默认 float64
pc2 = PointCloud.from_xyz(xyz_f64)

print(f"点数: {pc.point_count()}")
```

!!! warning "属性方法要求 float32"
    `set_intensity`、`set_rgb`、`add_attribute`、`set_attribute` 等方法要求输入数组必须是 `dtype=float32`。

### 添加属性

点云支持多种属性：

```python
# 强度值（float32）
intensity = np.random.rand(10000).astype(np.float32) * 255
pc.set_intensity(intensity)

# RGB 颜色（3 个独立的 float32 通道）
r = np.random.rand(10000).astype(np.float32) * 255
g = np.random.rand(10000).astype(np.float32) * 255
b = np.random.rand(10000).astype(np.float32) * 255
pc.set_rgb(r, g, b)

# 自定义属性
custom = np.random.randn(10000).astype(np.float32)
pc.add_attribute("custom_feature", custom)
```

## 读取属性

使用对应的 getter 方法获取属性数组：

```python
# 获取坐标
xyz = pc.get_xyz()  # shape: (N, 3), dtype: float32

# 获取属性
if pc.has_intensity():
    intensity = pc.get_intensity()  # shape: (N,), dtype: float32

if pc.has_rgb():
    r, g, b = pc.get_rgb()  # 3 个 shape: (N,) 数组, dtype: uint8
```

## 体素下采样

体素下采样是最常用的点云降采样方法：

```python
from pcl_rustic import DownsampleStrategy

# 质心采样（推荐，保持几何精度）
pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.CENTROID
)

# 随机采样（速度最快）
pc_down_fast = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.RANDOM
)

print(f"原始: {pc.point_count():,}, 下采样: {pc_down.point_count():,}")
```

### 降采样策略

| 策略 | 说明 | 性能 | 适用场景 |
|------|------|------|----------|
| `CENTROID` | 选择最接近体素中心的点 | ⭐⭐⭐ | 通用，保持几何形状 |
| `RANDOM` | 选择体素中间位置的点 | ⭐⭐⭐⭐⭐ | 快速预览 |

## 坐标变换

### 平移

```python
translation = np.array([10.0, 20.0, 30.0], dtype=np.float32)
pc_translated = pc.rigid_transform(np.eye(3, dtype=np.float32), translation)
```

### 旋转

```python
# 3×3 旋转矩阵
rotation = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]
], dtype=np.float32)

pc_rotated = pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))
```

### 变换矩阵

```python
# 3×3 矩阵：旋转/缩放
pc_transformed = pc.transform(rotation_3x3)

# 4×4 矩阵：齐次坐标（旋转 + 平移）
transform_4x4 = np.eye(4, dtype=np.float32)
transform_4x4[:3, :3] = rotation
transform_4x4[:3, 3] = translation
pc_transformed = pc.transform(transform_4x4)
```

## 文件 I/O

### 读取点云

```python
from pcl_rustic import PointCloud

# LAZ/LAS 格式
pc = PointCloud.from_las("data/sample.laz")

# CSV 格式
pc = PointCloud.from_csv("data/sample.csv", delimiter=44, x="x", y="y", z="z")

# 自动格式检测
pc = PointCloud.load_from_file("data/sample.laz")
```

### 写入点云

```python
# LAZ（压缩）
pc.to_las("output/result.laz", compress=True)

# LAS（未压缩）
pc.to_las("output/result.las", compress=False)

# CSV
pc.to_csv("output/result.csv", delimiter=44, x="x", y="y", z="z")
```

## 点云信息

```python
count = pc.point_count()
xyz = pc.get_xyz()
min_bound = xyz.min(axis=0)
max_bound = xyz.max(axis=0)
center = (min_bound + max_bound) / 2

print(f"点数: {count:,}")
print(f"范围: {min_bound} - {max_bound}")
print(f"中心: {center}")
print(f"内存: {pc.memory_usage() / 1024 / 1024:.1f} MB")
```

## 性能提示

!!! tip "优化建议"
    1. **数据类型**: 使用 `float32` 而非 `float64` 以获得最佳性能
    2. **批量操作**: 一次性设置所有属性，避免多次调用
    3. **体素大小**: 选择合适的体素大小，过小会导致处理时间增加
    4. **内存管理**: 对于大点云，及时删除不需要的中间结果

```python
# ✅ 推荐：批量设置
xyz = np.random.randn(1000000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)
pc.set_intensity(intensity)
pc.set_rgb(r, g, b)

# ❌ 不推荐：多次小规模操作
for i in range(1000):
    xyz_chunk = np.random.randn(1000, 3).astype(np.float32)
    pc_chunk = PointCloud.from_xyz(xyz_chunk)
```

## 下一步

- [更多示例](examples.md) - 查看完整的应用示例
- [API 文档](../api/overview.md) - 深入了解所有 API
- [性能基准](../performance/benchmarks.md) - 了解性能表现
