# 坐标变换

点云坐标系变换功能，提供矩阵变换与刚体变换。

## API 列表

- `PointCloud.transform(matrix: np.ndarray) -> PointCloud` - 应用 4x4 仿射变换矩阵
- `PointCloud.rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> PointCloud` - 应用旋转和平移

## 使用示例

### 平移变换

```python
import numpy as np
from pcl_rustic import PointCloud

# 创建点云
xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

# 平移 [x, y, z]
translation = np.array([10.0, 20.0, 30.0], dtype=np.float32)
pc_translated = pc.rigid_transform(np.eye(3, dtype=np.float32), translation)

# 验证平移
xyz_original = pc.get_xyz()
xyz_translated = pc_translated.get_xyz()
diff = xyz_translated.mean(axis=0) - xyz_original.mean(axis=0)
print(f"平移向量: {translation}")
print(f"实际位移: {diff}")
```

### 旋转变换

#### 绕 Z 轴旋转 90 度

```python
import numpy as np

# 90 度旋转矩阵（绕 Z 轴）
angle = np.pi / 2
rotation = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,              1]
], dtype=np.float32)

pc_rotated = pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))
```

#### 绕 X 轴旋转

```python
# 90 度旋转矩阵（绕 X 轴）
angle = np.pi / 2
rotation = np.array([
    [1, 0,              0             ],
    [0, np.cos(angle), -np.sin(angle)],
    [0, np.sin(angle),  np.cos(angle)]
], dtype=np.float32)

pc_rotated = pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))
```

#### 绕 Y 轴旋转

```python
# 90 度旋转矩阵（绕 Y 轴）
angle = np.pi / 2
rotation = np.array([
    [ np.cos(angle), 0, np.sin(angle)],
    [ 0,             1, 0            ],
    [-np.sin(angle), 0, np.cos(angle)]
], dtype=np.float32)

pc_rotated = pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))
```

### 组合变换

#### 先旋转后平移

```python
# 1. 旋转 45 度
angle = np.pi / 4
rotation = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,              1]
], dtype=np.float32)

pc_rotated = pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))

# 2. 平移
translation = np.array([10.0, 0.0, 0.0], dtype=np.float32)
pc_transformed = pc_rotated.rigid_transform(np.eye(3, dtype=np.float32), translation)
```

#### 使用 4x4 变换矩阵

```python
# 构建 4x4 变换矩阵（旋转 + 平移）
transform = np.eye(4, dtype=np.float32)

# 设置旋转部分 (3x3)
angle = np.pi / 4
transform[:3, :3] = np.array([
    [np.cos(angle), -np.sin(angle), 0],
    [np.sin(angle),  np.cos(angle), 0],
    [0,              0,              1]
], dtype=np.float32)

# 设置平移部分 (3x1)
transform[:3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)

# 应用变换
pc_transformed = pc.transform(transform)
```

## 相关链接

- [PointCloud](pointcloud.md) - 点云核心类
- [下采样](downsample.md) - 点云下采样
- [文件 I/O](io.md) - 读写文件
