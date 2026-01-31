# PointCloud 类

点云核心类，提供点云的创建、属性管理和基本操作功能。

!!! warning "数据类型要求"
    所有输入数组必须是 **`dtype=float32`** 的 NumPy 数组。

## API 列表

### 构造方法

- `PointCloud()`
- `PointCloud.from_xyz(xyz: np.ndarray) -> PointCloud`
- `PointCloud.from_xyz_intensity(xyz: np.ndarray, intensity: np.ndarray) -> PointCloud`
- `PointCloud.from_xyz_rgb(xyz: np.ndarray, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> PointCloud`
- `PointCloud.from_xyz_intensity_rgb(xyz: np.ndarray, intensity: np.ndarray, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> PointCloud`
- `PointCloud.from_dict(data: dict[str, np.ndarray]) -> PointCloud`

### 基本信息

- `point_count() -> int`
- `get_xyz() -> np.ndarray`
- `has_intensity() -> bool`
- `has_rgb() -> bool`
- `get_intensity() -> np.ndarray | None`
- `get_rgb() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None`

### 属性管理

- `set_intensity(intensity: np.ndarray) -> None`
- `set_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> None`
- `add_attribute(name: str, data: np.ndarray) -> None`
- `set_attribute(name: str, data: np.ndarray) -> None`
- `attribute_names() -> list[str]`
- `get_attribute(name: str) -> np.ndarray | None`
- `remove_attribute(name: str) -> None`
- `clear_attributes() -> None`
- `set_all_attributes(attributes: dict[str, list[float]]) -> None`
- `has_attributes(names: list[str]) -> bool`
- `attribute_info() -> list[tuple[str, int]]`
- `remove_intensity() -> None`
- `remove_rgb() -> None`

### 变换与下采样

- `transform(matrix: np.ndarray) -> PointCloud`
- `rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> PointCloud`
- `voxel_downsample(voxel_size: float, strategy: int = DownsampleStrategy.CENTROID) -> PointCloud`

### 文件 I/O

- `PointCloud.from_las(path: str) -> PointCloud`
- `to_las(path: str, compress: bool = False) -> None`
- `PointCloud.from_csv(...) -> PointCloud`
- `to_csv(...) -> None`
- `PointCloud.from_parquet(...) -> PointCloud`
- `to_parquet(...) -> None`
- `PointCloud.load_from_file(...) -> PointCloud`
- `save_to_file(...) -> None`

### 其他

- `memory_usage() -> int`
- `to_dict() -> dict[str, np.ndarray]`
- `clone() -> PointCloud`

## 使用示例

### 创建点云

```python
import numpy as np
from pcl_rustic import PointCloud

xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)
```

### 添加属性

```python
intensity = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_intensity(intensity)

r = np.random.rand(pc.point_count()).astype(np.float32) * 255
g = np.random.rand(pc.point_count()).astype(np.float32) * 255
b = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_rgb(r, g, b)
```

### 读取属性

```python
xyz = pc.get_xyz()

if pc.has_intensity():
    intensity = pc.get_intensity()

if pc.has_rgb():
    r, g, b = pc.get_rgb()
```

### 点云信息

```python
count = pc.point_count()
xyz = pc.get_xyz()
min_bound = xyz.min(axis=0)
max_bound = xyz.max(axis=0)
center = xyz.mean(axis=0)
```

## 相关链接

- [下采样](downsample.md) - 点云降采样方法
- [变换](transform.md) - 坐标系变换
- [示例](../getting-started/examples.md) - 更多使用示例# PointCloud 类

点云核心类，提供点云的创建、属性管理和基本操作功能。

!!! warning "数据类型要求"
    所有输入数组必须是 **`dtype=float32`** 的 NumPy 数组。

## API 列表

### 构造方法

- `PointCloud()`
```python
import numpy as np
from pcl_rustic import PointCloud

xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)
```

### 添加属性

```python
intensity = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_intensity(intensity)

r = np.random.rand(pc.point_count()).astype(np.float32) * 255
g = np.random.rand(pc.point_count()).astype(np.float32) * 255
b = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_rgb(r, g, b)
```

### 读取属性

```python
xyz = pc.get_xyz()

if pc.has_intensity():
    intensity = pc.get_intensity()

if pc.has_rgb():
    r, g, b = pc.get_rgb()
```

### 点云信息

```python
count = pc.point_count()
xyz = pc.get_xyz()
min_bound = xyz.min(axis=0)
max_bound = xyz.max(axis=0)
center = xyz.mean(axis=0)
```

## 相关链接

- [下采样](downsample.md) - 点云降采样方法
- [变换](transform.md) - 坐标系变换
- [示例](../getting-started/examples.md) - 更多使用示例
### 其他

- `memory_usage() -> int`
- `to_dict() -> dict[str, np.ndarray]`
- `clone() -> PointCloud`

## 使用示例

### 创建点云

```python
import numpy as np
from pcl_rustic import PointCloud

xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)
```

### 添加属性

```python
intensity = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_intensity(intensity)

r = np.random.rand(pc.point_count()).astype(np.float32) * 255
g = np.random.rand(pc.point_count()).astype(np.float32) * 255
b = np.random.rand(pc.point_count()).astype(np.float32) * 255
pc.set_rgb(r, g, b)
```

### 读取属性

```python
xyz = pc.get_xyz()

if pc.has_intensity():
    intensity = pc.get_intensity()

if pc.has_rgb():
    r, g, b = pc.get_rgb()
```

### 点云信息

```python
count = pc.point_count()
xyz = pc.get_xyz()
min_bound = xyz.min(axis=0)
max_bound = xyz.max(axis=0)
center = xyz.mean(axis=0)
```

## 相关链接

- [下采样](downsample.md) - 点云降采样方法
- [变换](transform.md) - 坐标系变换
- [示例](../getting-started/examples.md) - 更多使用示例
