# API 概览

本节提供 `pcl-rustic` 的完整 API 参考文档。

## 核心模块

pcl-rustic 提供以下主要模块和类：

### 点云类 (PointCloud)

[PointCloud](pointcloud.md) 是核心类，提供点云的创建、属性管理和基本操作功能。

**主要功能**：
- 从 NumPy 数组创建点云
- 属性管理（强度、RGB 颜色、自定义属性）
- 获取点云统计信息

### 下采样 (Downsample)

[下采样模块](downsample.md) 提供体素下采样功能，支持多种降采样策略。

**主要功能**：
- 体素下采样
- 2 种降采样策略：随机（RANDOM）、质心（CENTROID）

### 坐标变换 (Transform)

[变换模块](transform.md) 提供坐标系变换功能。

**主要功能**：
- 刚体变换（旋转 + 平移）
- 矩阵变换（支持 3×3 和 4×4）

### 文件 I/O

[I/O 模块](io.md) 提供多格式点云文件的读写功能。

**主要功能**：
- LAZ/LAS 文件读写
- CSV 文件读写
- Parquet 文件读写
- 自动格式检测（load_from_file / save_to_file）

## 数据类型要求

### 输入数据

`from_xyz` 支持多种 NumPy dtype：

| dtype | 支持 | 说明 |
|-------|------|------|
| `float32` | ✅ | 推荐，零拷贝 |
| `float64` | ✅ | 自动转换 |
| `int32` | ✅ | 自动转换 |
| `int64` | ✅ | 自动转换 |

!!! warning "自定义属性要求"
    `set_intensity`、`set_rgb`、`add_attribute`、`set_attribute` 等方法的输入数组必须是 **`dtype=float32`**。

### 输出数据

- `get_xyz()` 返回 `dtype=float32` 的 NumPy 数组（shape `[N, 3]`）
- `get_intensity()` 返回 `dtype=float32` 的 NumPy 数组（shape `[N]`）
- `get_rgb()` 返回 3 个 `dtype=uint8` 的 NumPy 数组（shape `[N]`）

## 快速索引

| 类/函数 | 描述 | 链接 |
|---------|------|------|
| `PointCloud` | 核心点云类 | [详情](pointcloud.md) |
| `PointCloud.from_xyz()` | 从 XYZ 数组创建 | [详情](pointcloud.md) |
| `PointCloud.from_las()` | 读取 LAZ/LAS 文件 | [详情](io.md) |
| `PointCloud.from_csv()` | 读取 CSV 文件 | [详情](io.md) |
| `PointCloud.from_parquet()` | 读取 Parquet 文件 | [详情](io.md) |
| `voxel_downsample()` | 体素下采样 | [详情](downsample.md) |
| `transform()` | 矩阵变换（3×3 / 4×4） | [详情](transform.md) |
| `rigid_transform()` | 刚体变换 | [详情](transform.md) |
| `to_las()` | 写入 LAZ/LAS 文件 | [详情](io.md) |
| `DownsampleStrategy.RANDOM` | 随机采样策略 | [详情](downsample.md) |
| `DownsampleStrategy.CENTROID` | 质心采样策略 | [详情](downsample.md) |

## 使用示例

### 基本工作流

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

# 1. 创建点云
xyz = np.random.randn(10000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)

# 2. 添加属性
intensity = np.random.rand(10000).astype(np.float32) * 255
pc.set_intensity(intensity)

# 3. 下采样
pc_down = pc.voxel_downsample(0.15, DownsampleStrategy.CENTROID)

# 4. 变换
translation = np.array([10.0, 0.0, 0.0], dtype=np.float32)
pc_translated = pc_down.rigid_transform(np.eye(3, dtype=np.float32), translation)

# 5. 保存
pc_translated.to_las("output.laz", compress=True)
```

## 设计原则

本 API 遵循以下设计原则：

1. **类型安全**：使用 `.pyi` 存根文件提供完整的类型注解
2. **零拷贝**：NumPy 数组与 Rust 张量之间尽可能避免数据拷贝
3. **批量操作**：所有操作都针对批量数据优化，不支持单点访问
4. **明确错误**：提供清晰的中文错误消息

## 下一步

- [点云类完整 API](pointcloud.md)
- [下采样 API](downsample.md)
- [坐标变换 API](transform.md)
- [文件 I/O API](io.md)
