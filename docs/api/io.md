# 文件 I/O

点云文件读写功能，支持 LAZ/LAS、CSV 和 Parquet 格式。

## API 列表

### LAZ/LAS 格式

| 方法 | 说明 |
|------|------|
| `PointCloud.from_las(path)` | 从 LAS/LAZ 文件读取点云 |
| `to_las(path, compress=False)` | 将点云写入 LAS（compress=False）或 LAZ（compress=True）文件 |
| `PointCloud.delete_file(path)` | 删除文件 |

### CSV 格式

| 方法 | 说明 |
|------|------|
| `PointCloud.from_csv(path, delimiter=44, x=None, y=None, ...)` | 从 CSV 文件读取 |
| `to_csv(path, delimiter=44, x=None, y=None, ...)` | 写入 CSV 文件 |

### Parquet 格式

| 方法 | 说明 |
|------|------|
| `PointCloud.from_parquet(path, x=None, y=None, ...)` | 从 Parquet 文件读取 |
| `to_parquet(path, x=None, y=None, ...)` | 写入 Parquet 文件 |

### 通用接口

| 方法 | 说明 |
|------|------|
| `PointCloud.load_from_file(path, x=None, ...)` | 自动检测格式并读取 |
| `save_to_file(path, x=None, ...)` | 根据文件扩展名自动选择格式 |

### 列名参数

CSV、Parquet 和通用接口都接受以下可选列名参数（均为 `str | None`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `x` | `None` | X 坐标列名 |
| `y` | `None` | Y 坐标列名 |
| `z` | `None` | Z 坐标列名 |
| `intensity` | `None` | intensity 列名 |
| `rgb_r` | `None` | R 通道列名 |
| `rgb_g` | `None` | G 通道列名 |
| `rgb_b` | `None` | B 通道列名 |

!!! note "delimiter 参数"
    CSV 格式的 `delimiter` 参数是**字节的 ASCII 码**（整数），不是字符：
    - 逗号: `44`（即 `b','`）
    - 制表符: `9`（即 `b'\t'`）

## 使用示例

### LAZ/LAS 基本读写

```python
from pcl_rustic import PointCloud

# 读取 LAZ 文件
pc = PointCloud.from_las("input.laz")
print(f"读取了 {pc.point_count():,} 个点")

# 处理点云
pc_down = pc.voxel_downsample(0.15)

# 保存为 LAZ（压缩）
pc_down.to_las("output.laz", compress=True)

# 保存为 LAS（未压缩）
pc_down.to_las("output.las", compress=False)
```

### CSV 读写

```python
from pcl_rustic import PointCloud

# 从 CSV 读取（逗号分隔，delimiter=44）
pc = PointCloud.from_csv(
    "input.csv",
    delimiter=44,
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)

# 写入 CSV
pc.to_csv(
    "output.csv",
    delimiter=44,
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)
```

### Parquet 读写

```python
from pcl_rustic import PointCloud

# 从 Parquet 读取
pc = PointCloud.from_parquet(
    "input.parquet",
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)

# 写入 Parquet
pc.to_parquet(
    "output.parquet",
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)
```

### 自动格式检测

```python
from pcl_rustic import PointCloud

# 根据文件扩展名自动检测格式
pc = PointCloud.load_from_file(
    "data.laz",
    x="x",
    y="y",
    z="z",
)

# 自动根据扩展名选择格式
pc.save_to_file(
    "output.parquet",
    x="x",
    y="y",
    z="z",
)
```

### 文件格式对比

```python
# LAS：未压缩，文件较大
pc.to_las("output.las", compress=False)

# LAZ：压缩格式，文件较小
pc.to_las("output.laz", compress=True)

# CSV：文本格式，易于处理，文件最大
pc.to_csv("output.csv", delimiter=44, x="x", y="y", z="z")

# Parquet：列式存储，高效处理，适合大规模数据
pc.to_parquet("output.parquet", x="x", y="y", z="z")
```

## 文件格式选择指南

| 格式 | 压缩 | 文件大小 | 读写速度 | 适用场景 |
|-----|------|---------|---------|---------|
| LAS | ❌ | 很大 | 快 | 行业标准，兼容性好 |
| LAZ | ✅ | 小 | 中等 | 存储、传输 |
| CSV | ❌ | 很大 | 慢 | 数据交换、人工检查 |
| Parquet | ✅ | 小 | 快 | 大规模数据、分析 |

## 属性处理

### 添加自定义属性后保存

```python
from pcl_rustic import PointCloud
import numpy as np

# 创建点云
xyz = np.random.randn(1000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)

# 添加属性（必须是 float32）
intensity = np.random.rand(1000).astype(np.float32) * 255
pc.set_intensity(intensity)

custom_attr = np.random.randn(1000).astype(np.float32)
pc.add_attribute("custom", custom_attr)

# 保存到 Parquet（保留所有属性）
pc.to_parquet("output.parquet", x="x", y="y", z="z", intensity="intensity")
```

## 相关链接

- [PointCloud](pointcloud.md) - 点云核心类
- [下采样](downsample.md) - 点云下采样
- [变换](transform.md) - 坐标变换
