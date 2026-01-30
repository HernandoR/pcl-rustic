# PCL Rustic 快速参考

## 安装与构建

```bash
# 构建wheel包
uv build

# 开发模式安装
maturin develop

# 运行测试
pytest tests/ -v

# 构建文档
cargo doc --open
```

## 核心API速查

### 创建点云

```python
from pcl_rustic import PointCloud

# 空点云
pc = PointCloud()

# 从XYZ创建
pc = PointCloud.from_xyz([[1,2,3], [4,5,6]])

# 从文件读取
pc = PointCloud.from_las("data.las")
pc = PointCloud.from_csv("data.csv", delimiter=ord(','))
```

### 属性操作

```python
# 设置属性（覆盖）
pc.set_intensity([100.0, 200.0])
pc.set_rgb([[255,0,0], [0,255,0]])

# 添加自定义属性（重复报错）
pc.add_attribute("confidence", [0.9, 0.8])

# 设置自定义属性（覆盖）
pc.set_attribute("category", [1.0, 2.0])

# 获取属性
intensity = pc.get_intensity()
rgb = pc.get_rgb()
confidence = pc.get_attribute("confidence")

# 查询
has_int = pc.has_intensity()
has_rgb = pc.has_rgb()
attrs = pc.attribute_names()
```

### 坐标变换

```python
# 3x3矩阵变换（缩放/旋转）
matrix = [
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
]
pc_new = pc.transform(matrix)

# 4x4齐次坐标变换
matrix = [
    [1, 0, 0, 10],
    [0, 1, 0, 20],
    [0, 0, 1, 30],
    [0, 0, 0, 1],
]
pc_new = pc.transform(matrix)

# 刚体变换（旋转+平移）
rotation = [[1,0,0], [0,1,0], [0,0,1]]  # 恒等旋转
translation = [1.0, 2.0, 3.0]
pc_new = pc.rigid_transform(rotation, translation)
```

### 体素下采样

```python
from pcl_rustic import DownsampleStrategy

# 随机采样
pc_down = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM)

# 重心采样（推荐）
pc_down = pc.voxel_downsample(1.0, DownsampleStrategy.CENTROID)
```

### 文件I/O

```python
# 读取
pc = PointCloud.from_las("input.las")
pc = PointCloud.from_csv("input.csv", delimiter=ord(','))

# 写入
pc.to_las("output.las", compress=False)
pc.to_csv("output.csv", delimiter=ord(','))

# 删除
PointCloud.delete_file("file.las")
```

### 工具方法

```python
# 统计
count = pc.point_count()
memory = pc.memory_usage()  # 字节

# 克隆
pc2 = pc.clone()

# 打印
print(pc)  # PointCloud(points=100, intensity=Yes, rgb=No, attributes=2)

# 转numpy字典
data = pc.to_dict()  # {'xyz': array, 'intensity': array, ...}
```

## 模块结构

```
src/
├── lib.rs                    # PyO3模块入口 + Python绑定
├── traits/                   # 接口定义
│   ├── point_cloud.rs       # PointCloudCore, PointCloudProperties
│   ├── io.rs                # IOConvert
│   ├── downsample.rs        # VoxelDownsample, DownsampleStrategy
│   └── transform.rs         # CoordinateTransform
├── point_cloud/              # 核心实现
│   ├── core.rs              # HighPerformancePointCloud
│   ├── attributes.rs        # 属性管理
│   ├── transform.rs         # 变换实现
│   └── voxel.rs             # 下采样 + 策略
├── io/                       # 多格式I/O
│   ├── las_laz.rs           # LAS/LAZ格式
│   ├── parquet.rs           # Parquet格式
│   └── csv.rs               # CSV格式
├── interop/                  # 跨生态互通
│   └── numpy.rs             # numpy转换
└── utils/                    # 工具
    ├── error.rs             # 异常处理
    ├── tensor.rs            # 张量验证
    └── reflect.rs           # 体素分组
```

## Trait体系

| Trait | 用途 | 实现者 |
|-------|------|--------|
| `PointCloudCore` | 读取点云基本数据 | `HighPerformancePointCloud` |
| `PointCloudProperties` | 修改点云属性 | `HighPerformancePointCloud` |
| `CoordinateTransform` | 坐标变换 | `HighPerformancePointCloud` |
| `VoxelDownsample` | 体素下采样 | `HighPerformancePointCloud` |
| `DownsampleStrategy` | 采样策略 | `RandomSampleStrategy`, `CentroidSampleStrategy` |
| `IOConvert` | 多格式I/O | `HighPerformancePointCloud` |

## 数据结构

```rust
pub struct HighPerformancePointCloud {
    xyz: Vec<Vec<f32>>,                    // [M, 3]
    intensity: Option<Vec<f32>>,           // [M]
    rgb: Option<Vec<Vec<u8>>>,             // [M, 3]
    attributes: HashMap<String, Vec<f32>>, // [M]
}
```

**约束**：
- 所有字段**私有**
- 仅支持**批量操作**，禁止单点访问
- Option类型避免不必要的内存占用

## 异常处理

| 异常类型 | 触发条件 |
|---------|---------|
| `ValueError` | 维度不匹配、无效参数、重复属性 |
| `IOError` | 文件读写错误 |
| `FileNotFoundError` | 文件不存在 |
| `MemoryError` | 内存不足 |

```python
try:
    pc = PointCloud.from_las("nonexistent.las")
except FileNotFoundError as e:
    print(f"文件错误：{e}")
except ValueError as e:
    print(f"参数错误：{e}")
```

## 扩展示例

### 添加新采样策略

```rust
// src/point_cloud/voxel.rs
pub struct MaxIntensitySampleStrategy;

impl DownsampleStrategy for MaxIntensitySampleStrategy {
    fn select_representative(
        &self,
        indices: Vec<usize>,
        _xyz: &[Vec<f32>],
    ) -> Result<usize> {
        Ok(indices[0])  // 实现你的逻辑
    }
    fn name(&self) -> &str { "MaxIntensity" }
}
```

### 添加新文件格式

```rust
// src/io/ply.rs
pub fn from_ply(path: &str) -> Result<HighPerformancePointCloud> {
    // 实现PLY读取
}

pub fn to_ply(pc: &HighPerformancePointCloud, path: &str) -> Result<()> {
    // 实现PLY写入
}
```

## 性能优化建议

| 优化方向 | 方法 |
|---------|------|
| 内存 | 使用Option类型，避免不必要的属性 |
| 速度 | 增加voxel_size，减少点数 |
| I/O | 使用LAZ压缩，批量处理 |
| 计算 | 利用burn张量的并行化 |

## 常用命令

```bash
# Rust编码规范
cargo fmt          # 格式化
cargo clippy       # 检查风格
cargo test         # 运行单元测试
cargo build --release  # 发布构建

# Python编码规范
black .            # 代码格式化
isort .            # 导入排序
mypy .             # 类型检查
pytest tests/      # 运行测试

# 文档
cargo doc --open   # 生成Rust文档
```

## 调试技巧

```python
# 打印点云信息
print(f"点数: {pc.point_count()}")
print(f"内存: {pc.memory_usage()} 字节")
print(f"XYZ范围: {min(pc.get_xyz())}-{max(pc.get_xyz())}")

# 检查属性
if pc.has_intensity():
    print(f"Intensity: {pc.get_intensity()}")
print(f"自定义属性: {pc.attribute_names()}")

# 检查变换效果
print(f"变换前: {pc.get_xyz()}")
pc_new = pc.transform(matrix)
print(f"变换后: {pc_new.get_xyz()}")
```

## 关键点

✅ **必须记住的**：
- 所有操作都是**批量的**，无单点访问
- 属性维度必须与点数**一致**
- 矩阵必须是**3x3或4x4**
- 属性重复时`add`报错，`set`覆盖
- 下采样自动保留所有属性

⚠️ **常见错误**：
- 属性长度不匹配 → ValueError
- 重复添加属性 → ValueError
- 文件不存在 → FileNotFoundError
- 无效矩阵维度 → ValueError

📚 **更多信息**：
- 完整文档：见 [README.md](README.md)
- 开发指南：见 [DEVELOPMENT.md](DEVELOPMENT.md)
- 示例代码：见 [examples/](examples/)
- 测试用例：见 [tests/](tests/)
