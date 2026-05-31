# 性能优化指南

本指南提供深入的性能优化技巧和最佳实践。

## 数据类型优化

### 使用 float32

使用 `float32` 可以获得最佳性能：

**优势**：
- 内存占用减半（相比 float64）
- 缓存命中率更高
- SIMD 指令利用率更好
- 点云处理通常不需要 float64 精度

```python
# ✅ 推荐：直接使用 float32
xyz = np.random.randn(1000000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)

# 也支持：float64（自动转换，但有轻微开销）
xyz = np.random.randn(1000000, 3)  # float64
pc = PointCloud.from_xyz(xyz)  # 内部自动转换
```

### NumPy 数组布局

使用 C-连续数组以获得最佳性能：

```python
# ✅ 好：C-连续数组
xyz = np.ascontiguousarray(xyz)
pc = PointCloud.from_xyz(xyz)

# ❌ 差：非连续数组
xyz = xyz[::2]  # 步长切片导致非连续
pc = PointCloud.from_xyz(xyz)  # 需要内部复制
```

检查数组是否连续：

```python
if not xyz.flags['C_CONTIGUOUS']:
    xyz = np.ascontiguousarray(xyz)
```

## 体素下采样优化

### 选择合适的体素大小

体素大小应该根据点云密度和应用需求选择：

```python
def calculate_optimal_voxel_size(pc: PointCloud, target_ratio: float = 0.5):
    """计算最优体素大小以达到目标减少率"""
    xyz = pc.get_xyz()

    min_bound = xyz.min(axis=0)
    max_bound = xyz.max(axis=0)
    extent = max_bound - min_bound

    volume = np.prod(extent)
    density = pc.point_count() / max(volume, 1e-10)

    target_points = pc.point_count() * target_ratio
    target_voxel_volume = volume / max(target_points, 1)
    voxel_size = target_voxel_volume ** (1/3)

    return voxel_size

# 使用
voxel_size = calculate_optimal_voxel_size(pc, target_ratio=0.3)
pc_down = pc.voxel_downsample(voxel_size)
```

### 策略选择

根据应用场景选择降采样策略：

| 场景 | 策略 | 原因 |
|------|------|------|
| 可视化 | RANDOM | 速度最快，视觉效果足够 |
| 配准/重建 | CENTROID | 保持几何精度 |

```python
from pcl_rustic import DownsampleStrategy

# 快速预览
pc_preview = pc.voxel_downsample(0.2, DownsampleStrategy.RANDOM)

# 精确处理
pc_precise = pc.voxel_downsample(0.1, DownsampleStrategy.CENTROID)
```

## 内存管理

### 分块处理大文件

对于超大点云（> 1 亿点），考虑分块处理：

```python
def process_large_point_cloud(file_path: str, chunk_size: int = 10_000_000):
    """分块处理大点云"""
    results = []

    for chunk_xyz in read_chunks(file_path, chunk_size):
        pc_chunk = PointCloud.from_xyz(chunk_xyz)
        pc_down = pc_chunk.voxel_downsample(0.15)
        results.append(pc_down.get_xyz())

        del pc_chunk  # 显式释放内存

    merged_xyz = np.vstack(results)
    return PointCloud.from_xyz(merged_xyz)
```

### 及时释放中间结果

```python
# ✅ 好：及时释放
pc = PointCloud.from_las("large.laz")
pc_down = pc.voxel_downsample(0.15)
del pc  # 释放原始点云
pc_down.to_las("output.laz", compress=True)

# ❌ 差：保留多个副本
pc = PointCloud.from_las("large.laz")
pc1 = pc.voxel_downsample(0.10)
pc2 = pc.voxel_downsample(0.15)
pc3 = pc.voxel_downsample(0.20)
# 内存占用是单个副本的 4 倍
```

### 监控内存使用

```python
import psutil
import os

def get_memory_usage():
    """获取当前进程内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

mem_before = get_memory_usage()
pc = PointCloud.from_las("large.laz")
mem_after = get_memory_usage()
print(f"加载点云使用: {mem_after - mem_before:.1f} MB")
```

## 并行处理

### 使用 multiprocessing

对于多个独立的点云文件，使用多进程并行处理：

```python
from multiprocessing import Pool
from pathlib import Path

def process_single_file(file_path: str) -> tuple:
    """处理单个文件"""
    pc = PointCloud.from_las(file_path)
    pc_down = pc.voxel_downsample(0.15)

    output_path = file_path.replace(".laz", "_downsampled.laz")
    pc_down.to_las(output_path, compress=True)

    return (file_path, pc.point_count(), pc_down.point_count())

def batch_process(input_dir: str, n_workers: int = 4):
    """批量并行处理"""
    files = list(Path(input_dir).glob("*.laz"))

    with Pool(n_workers) as pool:
        results = pool.map(process_single_file, [str(f) for f in files])

    for file_path, original, downsampled in results:
        reduction = (1 - downsampled / original) * 100
        print(f"{Path(file_path).name}: {original:,} → {downsampled:,} ({reduction:.1f}%)")

# 使用
batch_process("data/raw", n_workers=8)
```

### 避免过度并行

```python
import os

# ✅ 好：根据 CPU 核心数选择
n_workers = min(os.cpu_count(), len(files))

# ❌ 差：过多进程导致资源竞争
n_workers = 32  # 在 8 核 CPU 上
```

## I/O 优化

### 使用 LAZ 而非 LAS

LAZ 文件通常比 LAS 小 7–10 倍，读取速度相似：

```python
# ✅ 推荐：使用压缩格式
pc.to_las("output.laz", compress=True)  # ~100 MB

# ❌ 不推荐：未压缩
pc.to_las("output.las", compress=False)  # ~700 MB
```

## 算法优化

### 空间索引

对于多次查询，考虑构建空间索引：

```python
from scipy.spatial import cKDTree

xyz = pc.get_xyz()
tree = cKDTree(xyz)

# 半径搜索
indices = tree.query_ball_point([0, 0, 0], r=10.0)
nearby_points = xyz[indices]

# K 近邻搜索
distances, indices = tree.query([0, 0, 0], k=100)
```

### 预计算常用属性

```python
class CachedPointCloud:
    """带缓存的点云类"""
    def __init__(self, pc: PointCloud):
        self.pc = pc
        self._xyz = None
        self._center = None
        self._bounds = None

    @property
    def xyz(self):
        if self._xyz is None:
            self._xyz = self.pc.get_xyz()
        return self._xyz

    @property
    def center(self):
        if self._center is None:
            self._center = self.xyz.mean(axis=0)
        return self._center

    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = (self.xyz.min(axis=0), self.xyz.max(axis=0))
        return self._bounds
```

## 性能分析

### 使用 cProfile

```python
import cProfile
import pstats

def profile_function():
    pc = PointCloud.from_las("data.laz")
    pc_down = pc.voxel_downsample(0.15)
    pc_down.to_las("output.laz", compress=True)

cProfile.run('profile_function()', 'profile_stats')

stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### 使用 line_profiler

```python
from line_profiler import LineProfiler

def process_point_cloud(file_path: str):
    pc = PointCloud.from_las(file_path)
    pc_down = pc.voxel_downsample(0.15)
    pc_down.to_las("output.laz", compress=True)
    return pc_down

lp = LineProfiler()
lp.add_function(process_point_cloud)
lp.run('process_point_cloud("data.laz")')
lp.print_stats()
```

### 使用 memory_profiler

```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    pc = PointCloud.from_las("large.laz")
    pc_down = pc.voxel_downsample(0.15)
    return pc_down

memory_intensive_operation()
```

## 常见性能陷阱

### 1. 重复获取数据

```python
# ❌ 差：重复获取
for i in range(100):
    xyz_i = pc.get_xyz()  # 每次都分配新数组

# ✅ 好：缓存结果
xyz = pc.get_xyz()
for i in range(100):
    # 使用缓存的 xyz
    pass
```

### 2. 不必要的拷贝

```python
# ❌ 差：创建不必要的副本
xyz_copy = xyz.copy()
pc = PointCloud.from_xyz(xyz_copy)

# ✅ 好：直接使用
pc = PointCloud.from_xyz(xyz)
```

### 3. 过小的体素

```python
# ❌ 差：体素过小导致输出点数过多
pc_down = pc.voxel_downsample(0.001)  # 几乎没有减少

# ✅ 好：选择合理的体素大小
pc_down = pc.voxel_downsample(0.15)
```

## 性能检查清单

在部署到生产环境前，检查以下项目：

- [ ] 使用 `float32` 而非 `float64`
- [ ] 数组是 C-连续的
- [ ] 选择了合适的体素大小
- [ ] 选择了合适的降采样策略
- [ ] 及时释放大对象
- [ ] 使用 LAZ 而非 LAS
- [ ] 考虑了并行处理
- [ ] 缓存了常用属性
- [ ] 进行了性能分析
- [ ] 监控了内存使用

## 性能目标

根据我们的基准测试，以下是合理的性能目标：

| 操作 | 目标性能 |
|------|---------|
| 体素下采样 (10M) | < 10 秒 |
| 体素下采样 (50M) | < 50 秒 |
| 读取 LAZ (1GB) | < 30 秒 |
| 写入 LAZ (1GB) | < 40 秒 |
| 坐标变换 (10M) | < 1 秒 |

如果你的性能低于这些目标，请参考本指南进行优化。

## 下一步

- [基准测试](benchmarks.md) - 查看详细的性能数据
- [API 文档](../api/overview.md) - 了解 API 细节
- [开发指南](../development/contributing.md) - 参与性能优化
