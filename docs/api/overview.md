# API 概览

本节提供 `pcl-rustic` 的完整 API 参考文档，文档由手工维护并与当前接口保持一致。

## 核心模块

pcl-rustic 提供以下主要模块和类：

### 点云类 (PointCloud)

[PointCloud](pointcloud.md) 是核心类，提供点云的创建、属性管理和基本操作功能。

**主要功能**：
- 从 NumPy 数组创建点云
- 属性管理（强度、RGB 颜色等）
- 获取点云统计信息

### 下采样 (Downsample)

[下采样模块](downsample.md) 提供体素下采样功能，支持多种降采样策略。

**主要功能**：
- 体素下采样
- 多种降采样策略（质心、随机、强度加权质心）

### 坐标变换 (Transform)

[变换模块](transform.md) 提供坐标系变换功能。

**主要功能**：
- 平移变换
- 旋转变换
- 仿射变换

### 文件 I/O

[I/O 模块](io.md) 提供多格式点云文件的读写功能。

**主要功能**：
- LAZ/LAS 文件读写
- CSV 文件读写
- Parquet 文件读写（规划中）

## 数据类型要求

!!! warning "重要"
    所有输入的 NumPy 数组必须是 **`dtype=float32`**。如果数据是其他类型，需要使用 `.astype(np.float32)` 转换。

## 快速索引

| 类/函数 | 描述 | 链接 |
|---------|------|------|
| `PointCloud` | 核心点云类 | [详情](pointcloud.md) |
| `PointCloud.from_xyz()` | 从 XYZ 数组创建 | [详情](pointcloud.md) |
| `voxel_downsample()` | 体素下采样 | [详情](downsample.md) |
| `transform()` | 矩阵变换 | [详情](transform.md) |
| `rigid_transform()` | 刚体变换 | [详情](transform.md) |
| `from_las()` | 读取 LAZ/LAS 文件 | [详情](io.md) |
| `to_las()` | 写入 LAZ/LAS 文件 | [详情](io.md) |

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
