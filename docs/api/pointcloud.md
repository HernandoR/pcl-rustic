# PointCloud 类

点云核心类，提供点云的创建、属性管理和基本操作功能。

## 数据类型说明

| 方法 | 输入 dtype | 输出 dtype |
|------|-----------|-----------|
| `from_xyz()` | float32/float64/int32/int64 | — |
| `set_intensity()` | float32 | — |
| `set_rgb()` | float32 | — |
| `add_attribute()` / `set_attribute()` | float32 | — |
| `get_xyz()` | — | float32 |
| `get_intensity()` | — | float32 |
| `get_rgb()` | — | uint8 |

## API 列表

### 构造方法

| 方法 | 说明 |
|------|------|
| `PointCloud()` | 创建空点云 |
| `PointCloud.from_xyz(xyz)` | 从 `[N, 3]` XYZ 数组创建 |
| `PointCloud.from_xyz_intensity(xyz, intensity)` | 从 XYZ + intensity 创建 |
| `PointCloud.from_xyz_rgb(xyz, r, g, b)` | 从 XYZ + RGB 创建 |
| `PointCloud.from_xyz_intensity_rgb(xyz, intensity, r, g, b)` | 从 XYZ + intensity + RGB 创建 |
| `PointCloud.from_dict(data)` | 从字典创建（key: `xyz`, `intensity`, `r`, `g`, `b`） |

### 基本信息

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `point_count()` | `int` | 获取点数 |
| `get_xyz()` | `np.ndarray` | 获取 XYZ 坐标，shape `[N, 3]`，dtype float32 |
| `has_intensity()` | `bool` | 检查是否有 intensity |
| `has_rgb()` | `bool` | 检查是否有 RGB |
| `get_intensity()` | `np.ndarray \| None` | 获取 intensity，shape `[N]`，dtype float32 |
| `get_rgb()` | `tuple[np.ndarray, np.ndarray, np.ndarray] \| None` | 获取 RGB 三通道，dtype uint8 |
| `attribute_names()` | `list[str]` | 获取自定义属性名列表 |
| `get_attribute(name)` | `np.ndarray \| None` | 获取指定属性 |
| `attribute_info()` | `list[tuple[str, int]]` | 获取所有属性名和长度 |
| `has_attributes(names)` | `bool` | 检查是否包含所有指定属性 |
| `memory_usage()` | `int` | 估算内存占用（字节） |

### 属性管理

| 方法 | 说明 |
|------|------|
| `set_intensity(intensity)` | 设置 intensity（float32 数组） |
| `set_rgb(r, g, b)` | 设置 RGB（3 个 float32 数组） |
| `add_attribute(name, data)` | 添加自定义属性（重复时报错） |
| `set_attribute(name, data)` | 设置自定义属性（重复时覆盖） |
| `remove_attribute(name)` | 删除自定义属性 |
| `clear_attributes()` | 清除所有自定义属性 |
| `set_all_attributes(attributes)` | 批量设置所有自定义属性 |
| `remove_intensity()` | 移除 intensity |
| `remove_rgb()` | 移除 RGB |

### 变换与下采样

| 方法 | 说明 |
|------|------|
| `transform(matrix)` | 矩阵变换（支持 3×3 或 4×4） |
| `rigid_transform(rotation, translation)` | 刚体变换（3×3 旋转 + 3 维平移） |
| `voxel_downsample(voxel_size, strategy)` | 体素下采样 |

### 文件 I/O

| 方法 | 说明 |
|------|------|
| `PointCloud.from_las(path)` | 从 LAZ/LAS 文件读取 |
| `to_las(path, compress=False)` | 写入 LAZ/LAS 文件 |
| `PointCloud.from_csv(path, ...)` | 从 CSV 文件读取 |
| `to_csv(path, ...)` | 写入 CSV 文件 |
| `PointCloud.from_parquet(path, ...)` | 从 Parquet 文件读取 |
| `to_parquet(path, ...)` | 写入 Parquet 文件 |
| `PointCloud.load_from_file(path, ...)` | 自动检测格式并读取 |
| `save_to_file(path, ...)` | 根据扩展名自动选择格式写入 |
| `PointCloud.delete_file(path)` | 删除文件 |

### 其他

| 方法 | 说明 |
|------|------|
| `to_dict()` | 转换为 NumPy 数组字典 |
| `clone()` | 深拷贝点云 |
| `__repr__()` | 可读的字符串表示 |

## 使用示例

### 创建点云

```python
import numpy as np
from pcl_rustic import PointCloud

# 基础创建（推荐 float32）
xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

# 也支持其他 dtype（自动转换）
xyz_f64 = np.random.randn(10000, 3)  # float64
pc2 = PointCloud.from_xyz(xyz_f64)  # OK

# 带属性创建
intensity = np.random.rand(10000).astype(np.float32) * 255
pc3 = PointCloud.from_xyz_intensity(xyz, intensity)

# 从字典创建
data = {
    "xyz": xyz,
    "intensity": intensity,
}
pc4 = PointCloud.from_dict(data)
```

### 添加属性

```python
# intensity（float32）
intensity = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_intensity(intensity)

# RGB（3 个独立的 float32 通道）
r = np.full(pc.point_count(), 255, dtype=np.float32)
g = np.full(pc.point_count(), 128, dtype=np.float32)
b = np.full(pc.point_count(), 0, dtype=np.float32)
pc.set_rgb(r, g, b)

# 自定义属性
elevation = np.random.randn(pc.point_count()).astype(np.float32)
pc.add_attribute("elevation", elevation)
```

### 读取属性

```python
# XYZ 坐标（float32）
xyz = pc.get_xyz()  # shape: (N, 3)

# intensity（float32）
if pc.has_intensity():
    intensity = pc.get_intensity()  # shape: (N,)

# RGB（uint8）
if pc.has_rgb():
    r, g, b = pc.get_rgb()  # 各为 shape: (N,), dtype: uint8

# 自定义属性
elevation = pc.get_attribute("elevation")
```

### 点云信息

```python
count = pc.point_count()
xyz = pc.get_xyz()
min_bound = xyz.min(axis=0)
max_bound = xyz.max(axis=0)
center = xyz.mean(axis=0)

print(f"点数: {count:,}")
print(f"范围: {min_bound} ~ {max_bound}")
print(f"中心: {center}")
print(f"内存: {pc.memory_usage() / 1024 / 1024:.1f} MB")
```

### 属性检查

```python
# 检查特定属性是否存在
if pc.has_attributes(["intensity", "classification"]):
    print("包含所需属性")

# 查看所有属性信息
for name, length in pc.attribute_info():
    print(f"  {name}: {length} 个值")

# 列出属性名
print(pc.attribute_names())
```

## 相关链接

- [下采样](downsample.md) - 点云降采样方法
- [变换](transform.md) - 坐标系变换
- [文件 I/O](io.md) - 文件读写
- [示例](../getting-started/examples.md) - 更多使用示例
