# PCL Rustic — 高性能 Python 点云运算库

[![CI](https://github.com/YOUR_USERNAME/pcl-rustic/workflows/CI/badge.svg)](https://github.com/YOUR_USERNAME/pcl-rustic/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pcl-rustic?label=PyPI)](https://pypi.org/project/pcl-rustic/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange)](https://www.rust-lang.org/)
![License](https://img.shields.io/badge/license-MIT-green)

**PCL Rustic** 是一个基于 Rust + PyO3 的高性能 Python 点云处理库，使用 [Burn](https://github.com/tracel-ai/burn) 张量框架实现批量运算，支持 CPU/GPU 加速。

## ✨ 核心特性

- 🚀 **高性能批量运算** — 基于 Burn 张量框架，支持 CPU/GPU 加速
- 🔗 **NumPy 零拷贝互通** — 支持 float32/float64/int32/int64 多种 dtype
- 📦 **多格式 I/O** — LAZ/LAS/Parquet/CSV 格式读写，自动格式检测
- 🎯 **类型安全** — 完整的类型注解和 `.pyi` 存根文件
- 🧩 **模块化设计** — 清晰的 Trait 抽象，易于扩展
- 📊 **性能优异** — 10M 点云体素下采样 ~7s，吞吐量 1.3–1.5M pts/s

## 📦 安装

### 使用 uv (推荐)

```bash
uv pip install pcl-rustic
```

### 使用 pip

```bash
pip install pcl-rustic
```

### 从源码构建

需要 Python 3.10+ 和 Rust 1.70+

```bash
git clone https://github.com/YOUR_USERNAME/pcl-rustic.git
cd pcl-rustic
uv build
```

### 支持的 Python 版本

- Python 3.10 / 3.11 / 3.12 / 3.13
- Python 3.14t (free-threaded)

## 🚀 快速开始

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

# 创建点云（推荐 float32）
xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

# 添加属性
intensity = np.random.rand(10000).astype(np.float32) * 255
pc.set_intensity(intensity)

# 体素下采样
pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.CENTROID
)

print(f"原始: {pc.point_count():,} 点")
print(f"下采样: {pc_down.point_count():,} 点")
```

## 📖 API 文档

### 创建点云

```python
# 从 NumPy 数组创建（支持 float32/float64/int32/int64）
xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
pc = PointCloud.from_xyz(xyz)

# 从文件读取
pc = PointCloud.from_las("input.las")
pc = PointCloud.from_csv("input.csv", delimiter=44)
pc = PointCloud.from_parquet("input.parquet")
```

### 属性管理

```python
# 设置属性（dtype=float32）
intensity = np.array([100.0, 200.0], dtype=np.float32)
pc.set_intensity(intensity)

# 获取属性
xyz_arr = pc.get_xyz()          # shape: (N, 3), dtype: float32
intensity_arr = pc.get_intensity()  # shape: (N,), dtype: float32

# RGB（返回 uint8 数组）
if pc.has_rgb():
    r, g, b = pc.get_rgb()      # 各 shape: (N,), dtype: uint8

# 自定义属性
pc.add_attribute("elevation", data)
pc.set_attribute("classification", data)
pc.remove_attribute("elevation")
```

### 坐标变换

```python
# 刚体变换（3×3 旋转 + 3 维平移）
rotation = np.eye(3, dtype=np.float32)
translation = np.array([1.0, 2.0, 3.0], dtype=np.float32)
pc_t = pc.rigid_transform(rotation, translation)

# 矩阵变换（支持 3×3 和 4×4）
matrix = np.eye(4, dtype=np.float32)
pc_t = pc.transform(matrix)
```

### 下采样

```python
# 2 种采样策略
pc_down = pc.voxel_downsample(0.06, DownsampleStrategy.CENTROID)   # 质心
pc_down = pc.voxel_downsample(0.06, DownsampleStrategy.RANDOM)     # 随机
```

### 文件 I/O

```python
# LAZ/LAS
pc.to_las("output.las", compress=False)
pc.to_las("output.laz", compress=True)

# CSV（delimiter 为 ASCII 码: 44 = 逗号）
pc.to_csv("output.csv", delimiter=44)

# Parquet
pc.to_parquet("output.parquet")

# 自动格式检测
pc = PointCloud.load_from_file("data.laz")
pc.save_to_file("output.parquet")
```

## 🏗️ 架构设计

```
src/
├── lib.rs              # PyO3 Python 绑定入口
├── traits/             # Trait 抽象层
│   ├── point_cloud.rs  # PointCloudCore / PointCloudProperties
│   ├── downsample.rs   # DownsampleStrategy / VoxelDownsample
│   ├── transform.rs    # CoordinateTransform
│   └── io.rs           # I/O 接口
├── point_cloud/        # 点云核心实现
│   ├── core.rs         # HighPerformancePointCloud 结构体
│   ├── voxel.rs        # 体素下采样 + 采样策略
│   ├── transform.rs    # 坐标变换实现
│   └── attributes.rs   # 属性读写辅助
├── io/                 # 多格式 I/O
│   ├── las_laz.rs      # LAS/LAZ 格式
│   ├── csv.rs          # CSV 格式
│   ├── parquet.rs      # Parquet 格式
│   └── table.rs        # 表格列名解析
├── interop/            # Python 互通
│   └── numpy.rs        # NumPy 数组转换
└── utils/              # 工具模块
    ├── error.rs        # PointCloudError 错误处理
    ├── tensor.rs       # Burn 张量工具
    └── reflect.rs      # 反射/分组工具
```

**设计原则**：
- ✅ NumPy 数组作为 Python 接口（零拷贝读取）
- ✅ `from_xyz` 支持 float32/float64/int32/int64
- ✅ 属性方法（set_intensity/set_rgb/add_attribute）要求 float32
- ✅ `get_rgb()` 返回 uint8 数组
- ✅ 所有数据批量操作，不支持单点访问

## 🔧 开发指南

使用 [just](https://github.com/casey/just) 命令运行器简化开发工作流。

### 环境设置

```bash
just install    # 安装依赖 + pre-commit hooks
```

### 常用命令

```bash
just dev         # 开发模式构建
just build       # 生产模式构建
just test        # 运行测试
just test-fast   # 快速测试（跳过慢速）
just benchmark   # 性能基准测试
just fmt         # 格式化（cargo fmt + ruff format）
just lint        # Linting（cargo clippy + ruff check）
just pre-commit  # 运行所有 pre-commit hooks
just docs-serve  # 本地预览文档
just release     # 完整发布流程
just ci          # 模拟 CI 流程
just clean       # 清理构建产物
```

### 代码质量
- **Rust**: rustfmt + clippy
- **Python**: ruff (format + check)
- **Pre-commit**: 自动运行检查

### 性能基准

| 输入 | Voxel | 输出 | 减少率 | 耗时 | 吞吐量 |
|------|-------|------|-------|------|--------|
| 10M | 0.06 | 8.8M | 11.6% | 7.70s | 1.3M/s |
| 10M | 0.15 | 7.9M | 21.3% | 7.13s | 1.4M/s |
| 10M | 0.20 | 7.0M | 29.5% | 6.45s | 1.5M/s |
| 50M | 0.06 | 41.7M | 16.5% | 47.1s | 1.1M/s |
| 50M | 0.15 | 29.4M | 41.2% | 37.9s | 1.3M/s |
| 50M | 0.20 | 21.0M | 58.0% | 35.5s | 1.4M/s |

## 📊 数据格式要求

### from_xyz() — 支持多种 dtype

```python
# ✅ 推荐: float32
xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
pc = PointCloud.from_xyz(xyz)

# ✅ 支持: float64, int32, int64（自动转换）
xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
pc = PointCloud.from_xyz(xyz)
```

### 属性方法 — 必须 float32

```python
# ✅ 正确
intensity = np.array([100.0, 200.0], dtype=np.float32)
pc.set_intensity(intensity)

# ❌ 错误: float64
intensity = np.array([100.0, 200.0], dtype=np.float64)
pc.set_intensity(intensity)  # TypeError

# 修复: .astype(np.float32)
pc.set_intensity(intensity.astype(np.float32))
```

### 数据维度

- **XYZ**: `(N, 3)` 形状的 2D 数组
- **Intensity**: `(N,)` 形状的 1D 数组
- **自定义属性**: `(N,)` 形状的 1D 数组

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 确保通过所有检查:
   ```bash
   just fmt && just lint && just test && just pre-commit
   ```
4. 提交更改 (`git commit -m 'Add amazing feature'`)
5. 推送到分支 (`git push origin feature/amazing-feature`)
6. 创建 Pull Request

查看 [开发指南](https://YOUR_USERNAME.github.io/pcl-rustic/development/setup/) 了解更多。

## 📄 许可证

MIT License — 查看 [LICENSE](LICENSE) 文件。

## 👨‍💻 作者

**liuzhen19** — [liuzhen19@xiaomi.com](mailto:liuzhen19@xiaomi.com)

## 🔗 相关资源

- [Burn Framework](https://github.com/tracel-ai/burn) — Rust 深度学习框架
- [PyO3](https://pyo3.rs/) — Rust 的 Python 绑定
- [NumPy](https://numpy.org/) — Python 科学计算库
- [Maturin](https://github.com/PyO3/maturin) — Rust-Python 打包工具

## 🐛 问题排查

| 问题 | 解决 |
|------|------|
| `必须是dtype=float32的2D numpy数组` | `xyz = xyz.astype(np.float32)` |
| `error: failed to compile` | `rustup update stable && cargo clean && maturin develop --release` |
| `No module named 'pcl_rustic._core'` | `maturin develop --release` |

## 📈 路线图

- [ ] GPU 加速支持
- [ ] 更多下采样策略（FPS, Normal-based）
- [ ] 点云配准算法（ICP, NDT）
- [ ] 法向量估计
- [ ] 点云分割

---

**Star ⭐ 本项目以支持开发！**
