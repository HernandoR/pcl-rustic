# PCL Rustic 开发指南

## 项目设置

### 环境要求

- Rust 1.70+ (安装：https://rustup.rs/)
- Python 3.9+
- uv 包管理器 (安装：https://docs.astral.sh/uv/getting-started/)

### 初始化环境

```bash
# 1. 使用uv创建虚拟环境
uv venv

# 2. 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate       # Windows

# 3. 安装开发依赖
uv pip install -e ".[dev]"
```

## 构建过程

### 完整构建

```bash
# 使用maturin构建Python扩展
uv build

# 或者直接使用cargo
cargo build --release
```

### 开发模式编译

```bash
# 用于快速迭代开发
maturin develop

# 这会编译Rust代码并将其安装到当前虚拟环境中
```

### 构建wheel包

```bash
# 生成跨平台wheel包
uv build --wheel

# 输出文件位置：dist/pcl_rustic-*.whl
```

## 代码结构详解

### Rust模块组织

#### 1. traits模块（特征/接口）

**位置**：`src/traits/`

定义了所有的Trait接口，提供高级别的抽象：

```
traits/
├── mod.rs              # 模块入口
├── point_cloud.rs      # PointCloudCore, PointCloudProperties
├── io.rs              # IOConvert
├── downsample.rs      # VoxelDownsample, DownsampleStrategy
└── transform.rs       # CoordinateTransform
```

**设计原则**：
- 接口隔离：每个Trait只定义单一职责的方法
- 实现分离：Struct通过impl实现Trait，而不是在Struct定义中堆砌
- 扩展性：新的Struct通过实现Trait自动获得功能

#### 2. point_cloud模块（核心数据结构）

**位置**：`src/point_cloud/`

包含核心点云数据结构及其实现：

```
point_cloud/
├── mod.rs              # 模块入口，导出HighPerformancePointCloud
├── core.rs             # HighPerformancePointCloud结构体和基础方法
├── attributes.rs       # 属性管理方法
├── transform.rs        # CoordinateTransform的impl
└── voxel.rs           # VoxelDownsample的impl + 策略类
```

**关键设计**：
```rust
pub struct HighPerformancePointCloud {
    xyz: Vec<Vec<f32>>,                    // 必选：[M,3]
    intensity: Option<Vec<f32>>,           // 可选：[M]
    rgb: Option<Vec<Vec<u8>>>,             // 可选：[M,3]
    attributes: HashMap<String, Vec<f32>>, // 自定义属性
}
```

- 所有字段**私有**：通过Trait方法和公有接口暴露
- 使用Option避免内存浪费
- HashMap灵活支持任意自定义属性

#### 3. io模块（文件I/O）

**位置**：`src/io/`

实现多格式文件读写：

```
io/
├── mod.rs              # 模块入口
├── las_laz.rs          # LAS/LAZ格式（las crate）
├── parquet.rs          # Parquet格式（arrow-rs）
└── csv.rs             # CSV格式（csv crate）
```

**Trait统一接口**：所有格式通过`IOConvert` Trait实现统一的read/write/delete接口

#### 4. utils模块（工具函数）

**位置**：`src/utils/`

```
utils/
├── mod.rs              # 模块入口
├── error.rs            # 异常定义和Rust→Python转换
├── tensor.rs           # 张量验证和操作工具
└── reflect.rs          # 体素分组和采样工具
```

**error.rs关键函数**：
```rust
pub enum PointCloudError {
    IoError, ParseError, DimensionMismatch, ...
}

impl From<PointCloudError> for PyErr  // 自动转换为Python异常
```

**tensor.rs验证函数**：
- `validate_xyz_shape()` - 检查[M,3]形状
- `validate_rgb_shape()` - 检查[M,3]形状
- `validate_intensity_shape()` - 检查长度匹配

**reflect.rs分组函数**：
- `group_points_by_voxel()` - 体素量化分组
- `compute_voxel_centroid()` - 计算体素中心
- `find_closest_to_centroid()` - 找最近点

#### 5. interop模块（跨生态互通）

**位置**：`src/interop/`

支持与numpy/pandas等Python库的互操作：

```
interop/
├── mod.rs              # 模块入口
└── numpy.rs           # numpy数组转换
```

实现零拷贝或低拷贝的数据转换。

### Python绑定层

**位置**：`src/lib.rs`

使用PyO3将Rust接口暴露给Python：

```rust
#[pymodule]
fn _core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyDownsampleStrategy>()?;
    Ok(())
}
```

**PyPointCloud类**：
- Wraps `HighPerformancePointCloud`
- 提供Python友好的方法签名
- 自动异常转换

**PyDownsampleStrategy类**：
- Python枚举风格的策略常量
- RANDOM = 0, CENTROID = 1

## 工程实践

### 代码规范

#### Rust编码规范

```bash
# 格式化代码
cargo fmt

# 检查风格和常见错误
cargo clippy

# 运行单元测试
cargo test
```

**命名约定**：
- Struct：PascalCase (e.g., `HighPerformancePointCloud`)
- Trait：PascalCase (e.g., `PointCloudCore`)
- 函数：snake_case (e.g., `validate_xyz_shape`)
- 常量：UPPER_SNAKE_CASE (e.g., `DEFAULT_VOXEL_SIZE`)

#### Python编码规范

```bash
# 代码格式化
black .

# 导入排序
isort .

# 类型检查
mypy src/
```

### 扩展机制

#### 添加新的采样策略

**步骤1**：在`src/point_cloud/voxel.rs`定义策略

```rust
pub struct MyStrategy;

impl DownsampleStrategy for MyStrategy {
    fn select_representative(
        &self,
        indices: Vec<usize>,
        xyz: &[Vec<f32>],
    ) -> Result<usize> {
        // 实现逻辑
        Ok(indices[0])
    }

    fn name(&self) -> &str {
        "MyStrategy"
    }
}
```

**步骤2**：在`src/lib.rs`的PyO3绑定中映射策略ID

```rust
fn voxel_downsample(&self, voxel_size: f32, strategy: i32) -> PyResult<Self> {
    let strategy_impl: Box<dyn DownsampleStrategy> = match strategy {
        0 => Box::new(point_cloud::voxel::RandomSampleStrategy),
        1 => Box::new(point_cloud::voxel::CentroidSampleStrategy),
        2 => Box::new(point_cloud::voxel::MyStrategy),  // 新策略
        _ => return Err(PyValueError::new_err("未知策略")),
    };
    // ...
}
```

**步骤3**：在`src/pcl_rustic/_core.pyi`更新类型注解

```python
class DownsampleStrategy:
    RANDOM: int
    CENTROID: int
    MY_STRATEGY: int = 2  # 新常量
```

**步骤4**：在`src/lib.rs`添加Python类属性

```rust
#[pymethods]
impl PyDownsampleStrategy {
    #[classattr]
    fn MY_STRATEGY() -> i32 {
        2
    }
}
```

#### 添加新的文件格式

**步骤1**：创建新的IO模块

```bash
touch src/io/geojson.rs
```

**步骤2**：实现读写函数

```rust
// src/io/geojson.rs
pub fn from_geojson(path: &str) -> Result<HighPerformancePointCloud> {
    // 实现GeoJSON读取
}

pub fn to_geojson(pc: &HighPerformancePointCloud, path: &str) -> Result<()> {
    // 实现GeoJSON写入
}
```

**步骤3**：在`src/io/mod.rs`导出

```rust
pub mod geojson;
pub use geojson::*;
```

**步骤4**：在`HighPerformancePointCloud`添加方法

```rust
impl HighPerformancePointCloud {
    pub fn from_geojson(path: &str) -> Result<Self> {
        crate::io::geojson::from_geojson(path)
    }

    pub fn to_geojson(&self, path: &str) -> Result<()> {
        crate::io::geojson::to_geojson(self, path)
    }
}
```

## 测试策略

### 单元测试

在每个Rust模块中编写`#[cfg(test)]`块：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert_eq!(2 + 2, 4);
    }
}
```

运行测试：
```bash
cargo test
```

### 集成测试

使用pytest测试Python接口：

```bash
pytest tests/ -v
```

**测试分类**：
- `test_*_lifecycle.py` - 生命周期和创建
- `test_*_properties.py` - 属性操作
- `test_*_transform.py` - 坐标变换
- `test_*_downsample.py` - 下采样
- `test_*_io.py` - 文件I/O
- `test_*_integration.py` - 集成测试

### 性能基准测试

```python
import timeit

# 基准：点云创建
timer = timeit.Timer(
    'pc = PointCloud.from_xyz(xyz)',
    setup='from pcl_rustic import PointCloud; xyz = [[i,i,i] for i in range(10000)]'
)
print(f"创建10000点点云：{timer.timeit(100) / 100:.6f}秒/次")
```

## 调试指南

### Rust调试

**使用RUST_LOG环境变量**：
```bash
RUST_LOG=debug python example.py
```

**gdb调试**：
```bash
# 编译debug版本
cargo build

# 使用gdb
gdb python
(gdb) run example.py
```

### Python调试

**pdb调试**：
```python
import pdb
pdb.set_trace()

pc = PointCloud()
```

**使用IDE调试**：
- VS Code + Python extension
- PyCharm Professional

### 常见问题

| 问题 | 解决方案 |
|------|--------|
| `ImportError: No module named 'pcl_rustic._core'` | 运行 `maturin develop` 或 `uv build` |
| `PanicException: called Option::unwrap() on None` | 检查数据维度和长度匹配 |
| 内存占用过高 | 检查是否有属性泄露，使用`memory_usage()` |
| 下采样速度慢 | 调整voxel_size（更大更快），减少点数 |

## 文档维护

### 代码注释规范

```rust
/// 公开的函数/结构体说明
///
/// 更详细的说明，包括参数、返回值和异常
///
/// # Examples
/// ```
/// let result = my_function(vec![1.0, 2.0, 3.0]);
/// ```
pub fn my_function(data: Vec<f32>) -> Result<f32> {
    // 实现
}
```

### 文档生成

```bash
# 生成Rust API文档
cargo doc --open

# 生成Python文档（需要sphinx）
pip install sphinx
sphinx-build -b html docs docs/_build
```

## 版本管理

### 版本号格式

遵循语义化版本（Semantic Versioning）：
- `major.minor.patch` (e.g., `0.1.0`)
- Breaking changes → major+
- 新功能 → minor+
- Bug fix → patch+

### 更新版本

1. 更新`Cargo.toml`: `version = "0.2.0"`
2. 更新`pyproject.toml`: `version = "0.2.0"`
3. 更新`src/lib.rs`: 模块版本注释
4. 提交并标签：`git tag v0.2.0`

## CI/CD集成

### GitHub Actions示例

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v3
      - uses: PyO3/maturin-action@v1
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install pytest
      - run: pytest tests/
```

---

**最后更新**：2026年1月
