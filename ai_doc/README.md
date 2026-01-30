# PCL Rustic - 高性能Python点云运算库

![build](https://img.shields.io/badge/build-Rust%2BPyO3-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 简介

**PCL Rustic** 是一个基于Rust的高性能Python点云运算库，采用burn张量框架实现全批量张量运算。

### 核心特性

- **高性能批量运算**：基于burn张量库，支持CPU/GPU加速
- **多格式IO**：LAZ/LAS/Parquet/CSV格式读写删
- **零拷贝互通**：numpy数组无缝转换
- **模块化设计**：清晰的Trait抽象，易于扩展
- **Python友好**：符合PEP8规范，完整类型注解

## 架构概览

```
src/
├── lib.rs              # PyO3模块入口，Python绑定
├── traits/             # 通用Trait抽象
│   ├── point_cloud.rs  # 点云核心Trait
│   ├── io.rs           # IO接口Trait
│   ├── downsample.rs   # 下采样Trait
│   └── transform.rs    # 坐标变换Trait
├── point_cloud/        # 点云核心模块
│   ├── core.rs         # HighPerformancePointCloud结构体
│   ├── attributes.rs   # 属性管理方法
│   ├── transform.rs    # 变换实现
│   └── voxel.rs        # 下采样实现
├── io/                 # 多格式IO模块
│   ├── las_laz.rs      # LAS/LAZ格式
│   ├── parquet.rs      # Parquet格式
│   └── csv.rs          # CSV格式
├── interop/            # 跨生态互通
│   └── numpy.rs        # numpy数组转换
└── utils/              # 工具模块
    ├── error.rs        # 异常处理
    ├── tensor.rs       # 张量工具
    └── reflect.rs      # 反射和下采样工具
```

## 快速开始

### 安装

```bash
# 使用uv安装
uv pip install pcl-rustic

# 或者从源码构建
uv build
```

### 基本使用

```python
from pcl_rustic import PointCloud, DownsampleStrategy

# 创建点云
xyz = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
pc = PointCloud.from_xyz(xyz)

# 添加属性
pc.set_intensity([100.0, 200.0])
pc.set_rgb([[255, 0, 0], [0, 255, 0]])

# 坐标变换
rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
translation = [1.0, 2.0, 3.0]
pc_transformed = pc.rigid_transform(rotation, translation)

# 体素下采样
pc_downsampled = pc.voxel_downsample(
    voxel_size=1.0,
    strategy=DownsampleStrategy.CENTROID
)

print(f"原始点数: {pc.point_count()}")
print(f"下采样后点数: {pc_downsampled.point_count()}")
```

## API文档

### PointCloud 类

#### 创建

```python
# 空点云
pc = PointCloud()

# 从XYZ坐标创建
pc = PointCloud.from_xyz(xyz: List[List[float]]) -> PointCloud

# 从文件读取
pc = PointCloud.from_las(path: str) -> PointCloud
pc = PointCloud.from_csv(path: str, delimiter: int = ord(',')) -> PointCloud
```

#### 属性访问

```python
pc.point_count() -> int                          # 获取点数
pc.get_xyz() -> List[List[float]]                # 获取XYZ坐标
pc.has_intensity() -> bool                       # 检查是否有intensity
pc.get_intensity() -> Optional[List[float]]      # 获取intensity
pc.has_rgb() -> bool                             # 检查是否有RGB
pc.get_rgb() -> Optional[List[List[int]]]        # 获取RGB
```

#### 属性操作

```python
pc.set_intensity(data: List[float])              # 设置intensity（覆盖）
pc.set_rgb(data: List[List[int]])                # 设置RGB（覆盖）
pc.add_attribute(name: str, data: List[float])   # 添加属性（重复报错）
pc.set_attribute(name: str, data: List[float])   # 设置属性（覆盖）
pc.remove_attribute(name: str)                   # 删除属性
pc.attribute_names() -> List[str]                # 获取属性名列表
pc.get_attribute(name: str) -> Optional[List[float]]  # 获取属性
```

#### 坐标变换

```python
# 矩阵变换（3x3或4x4）
pc_new = pc.transform(matrix: List[List[float]]) -> PointCloud

# 刚体变换（旋转+平移）
pc_new = pc.rigid_transform(
    rotation: List[List[float]],
    translation: List[float]
) -> PointCloud
```

#### 下采样

```python
# 体素下采样
pc_downsampled = pc.voxel_downsample(
    voxel_size: float,
    strategy: int = DownsampleStrategy.CENTROID
) -> PointCloud
```

**采样策略**：
- `DownsampleStrategy.RANDOM`：随机选择体素内的点
- `DownsampleStrategy.CENTROID`：选择最接近体素中心的点

#### 文件I/O

```python
# 读取
pc = PointCloud.from_las(path: str) -> PointCloud
pc = PointCloud.from_csv(path: str, delimiter: int) -> PointCloud

# 写入
pc.to_las(path: str, compress: bool = False) -> None
pc.to_csv(path: str, delimiter: int = ord(',')) -> None

# 删除
PointCloud.delete_file(path: str) -> None
```

#### 其他方法

```python
pc.memory_usage() -> int                         # 获取内存占用（字节）
pc.to_dict() -> Dict[str, np.ndarray]           # 转换为numpy字典
pc.clone() -> PointCloud                         # 克隆点云
```

## 数据结构

### HighPerformancePointCloud

所有字段**私有**，通过Trait方法和公有接口暴露：

```rust
pub struct HighPerformancePointCloud {
    xyz: Vec<Vec<f32>>,                          // [M,3] 必选
    intensity: Option<Vec<f32>>,                 // [M,] 可选
    rgb: Option<Vec<Vec<u8>>>,                   // [M,3] 可选
    attributes: HashMap<String, Vec<f32>>,       // 自定义属性
}
```

**特点**：
- 连续存储，支持高效批量运算
- Option类型，无数据时不占用内存
- 不支持单点访问/设置，仅支持批量操作

## 开发指南

### 构建

```bash
# 完整构建（编译Rust + Python打包）
uv build

# 仅编译Rust
cargo build --release

# 运行测试
pytest tests/

# 构建wheel包
uv build --wheel
```

### 项目结构说明

#### Trait设计原则

1. **PointCloudCore**：点云基础能力（读取XYZ/intensity/RGB）
2. **PointCloudProperties**：属性管理（设置属性、自定义属性）
3. **CoordinateTransform**：坐标变换（矩阵/刚体变换）
4. **VoxelDownsample**：体素下采样（接收策略参数）
5. **DownsampleStrategy**：采样策略（接口实现）
6. **IOConvert**：多格式IO（统一读写删接口）

#### Class设计模式

**Rust侧**：
- 核心点云Struct（`HighPerformancePointCloud`）聚合所有数据
- 策略Struct（`RandomSampleStrategy`/`CentroidSampleStrategy`）实现接口
- 异常Struct（`PointCloudError`）统一异常处理

**Python侧**：
- 单核心Class（`PointCloud`）屏蔽Rust细节
- Enum策略（`DownsampleStrategy.RANDOM/CENTROID`）易于调用
- 完整类型注解（`.pyi`文件）支持mypy检查

### 扩展指南

#### 添加新的采样策略

1. 在 `src/point_cloud/voxel.rs` 中创建新Struct：

```rust
pub struct MyStrategy;

impl DownsampleStrategy for MyStrategy {
    fn select_representative(&self, indices: Vec<usize>, xyz: &[Vec<f32>]) -> Result<usize> {
        // 实现选择逻辑
        Ok(indices[0])
    }
    fn name(&self) -> &str {
        "MyStrategy"
    }
}
```

2. 在 `src/lib.rs` 的PyO3绑定中映射新策略的ID

3. 在Python中通过新ID调用

#### 添加新的文件格式

1. 在 `src/io/` 目录创建新模块（如 `geojson.rs`）
2. 实现读写函数
3. 在 `src/io/mod.rs` 中导出
4. 在 `HighPerformancePointCloud` 中实现相应方法

## 性能优化

### 内存优化

- 使用Option类型，避免不必要的内存分配
- 支持零拷贝numpy互通（使用ndarray视图）
- 连续存储优化缓存局部性

### 计算优化

- 全程基于burn张量，支持并行化
- 批量体素分组避免单点处理
- 适配CPU（必选）和GPU（可选）

## 测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 测试覆盖

- **核心功能**：生命周期、属性、变换、下采样
- **边界场景**：空点云、单点、大规模点云
- **异常场景**：维度不匹配、无效参数、文件错误
- **集成测试**：完整工作流、链式操作

## 贡献指南

欢迎提交Issue和PR！请遵循以下约定：

1. 代码遵循Rust编码规范（`cargo fmt`）
2. 添加单元测试（`#[cfg(test)]`）
3. Python代码遵循PEP8（`black`, `isort`）
4. 更新相关文档

## 许可证

MIT License

## 作者

liuzhen19 <liuzhen19@xiaomi.com>

## 相关资源

- [Burn Framework](https://github.com/tracel-ai/burn)
- [PyO3 Documentation](https://pyo3.rs/)
- [NumPy-Rust Integration](https://rust-numpy.github.io/)

## 问题排查

### 编译错误

**问题**：`pyo3` 版本冲突
```
error[E0308]: mismatched types...
```

**解决**：更新项目依赖
```bash
uv pip install --upgrade pyo3 maturin
```

### 导入错误

**问题**：`ModuleNotFoundError: No module named 'pcl_rustic._core'`

**解决**：确保已构建wheel包
```bash
uv build --wheel
uv pip install ./dist/pcl_rustic-*.whl
```

### 性能下降

**问题**：大规模点云处理缓慢

**解决**：
1. 检查点数规模和可用内存
2. 调整voxel_size（更大的值更快）
3. 考虑启用GPU后端（如果可用）

---

最后更新：2026年1月

## development

```bash
uv venv
uv sync --dev
maturin develop --uv
```