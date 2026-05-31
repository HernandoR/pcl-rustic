# 贡献指南

感谢您对 pcl-rustic 的贡献兴趣！本指南将帮助您了解如何参与项目开发。

## 行为准则

- 尊重所有贡献者
- 保持建设性的讨论
- 专注于技术问题
- 欢迎新手贡献者

## 如何贡献

### 报告 Bug

在 [GitHub Issues](https://github.com/YOUR_USERNAME/pcl-rustic/issues) 中创建 bug 报告，请包含：

- **描述**: 清晰简洁的问题描述
- **复现步骤**: 详细的复现步骤
- **期望行为**: 应该发生什么
- **实际行为**: 实际发生了什么
- **环境信息**: OS、Python 版本、pcl-rustic 版本
- **代码示例**: 最小可复现示例

**Bug 报告模板**:

```markdown
**描述**
简短描述 bug

**复现步骤**
1. ...
2. ...
3. ...

**期望行为**
应该...

**实际行为**
实际...

**环境**
- OS: macOS 14.0
- Python: 3.11
- pcl-rustic: 0.1.0

**代码示例**
```python
# 最小可复现示例
```
```

### 提议新功能

在 [GitHub Discussions](https://github.com/YOUR_USERNAME/pcl-rustic/discussions) 中讨论新功能：

- 描述功能的用途
- 解释为什么需要这个功能
- 提供使用示例
- 讨论可能的实现方式

### 贡献代码

1. **Fork 仓库**

   点击 GitHub 页面右上角的 "Fork" 按钮

2. **克隆你的 fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/pcl-rustic.git
   cd pcl-rustic
   ```

3. **创建特性分支**

   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **设置开发环境**

   ```bash
   just install
   ```

5. **进行更改**

   编辑代码，确保：
   - 代码符合项目规范
   - 添加了单元测试
   - 测试通过
   - 文档已更新

6. **运行测试**

   ```bash
   just test
   just fmt
   just lint
   ```

7. **提交更改**

   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   提交消息应遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：
   - `feat:` 新功能
   - `fix:` Bug 修复
   - `docs:` 文档更新
   - `style:` 代码格式（不影响功能）
   - `refactor:` 重构
   - `perf:` 性能优化
   - `test:` 测试相关
   - `chore:` 构建/工具相关

8. **推送到 GitHub**

   ```bash
   git push origin feature/amazing-feature
   ```

9. **创建 Pull Request**

   在 GitHub 上创建 PR，描述清楚：
   - 解决了什么问题
   - 如何解决的
   - 相关 Issue 编号
   - 测试情况

## 架构概览

```
src/
├── lib.rs              # PyO3 Python 绑定入口
├── traits/             # Trait 抽象层
│   ├── point_cloud.rs  # PointCloudCore / PointCloudProperties
│   ├── io.rs           # I/O 接口
│   ├── downsample.rs   # DownsampleStrategy / VoxelDownsample
│   └── transform.rs    # CoordinateTransform
├── point_cloud/        # 点云核心实现
│   ├── core.rs         # HighPerformancePointCloud 结构体
│   ├── voxel.rs        # 体素下采样实现 + 采样策略
│   ├── transform.rs    # 坐标变换实现
│   └── attributes.rs   # 属性读写辅助
├── io/                 # 多格式 I/O
│   ├── las_laz.rs      # LAS/LAZ 格式
│   ├── parquet.rs      # Parquet 格式
│   ├── csv.rs          # CSV 格式
│   └── table.rs        # 表格列名解析
├── interop/            # Python 互通
│   └── numpy.rs        # NumPy 数组转换
└── utils/              # 工具模块
    ├── error.rs        # 错误处理（PointCloudError）
    ├── tensor.rs       # Burn 张量工具
    └── reflect.rs      # 反射/分组工具
```

## 代码规范

### Rust 代码

- 遵循 Rust 标准风格（`cargo fmt`）
- 通过所有 clippy 检查（`cargo clippy`）
- 添加文档注释（`///`）
- 为公共 API 编写测试

**示例**:

```rust
/// 从 numpy XYZ 数组创建点云
///
/// # Arguments
/// * `xyz` - 形状为 [N, 3] 的 2D numpy 数组
///
/// # Returns
/// `PyResult<Self>` - 成功返回点云对象
#[staticmethod]
fn from_xyz(xyz: &Bound<'_, PyAny>) -> PyResult<Self> {
    // ...
}
```

### Python 代码

- 遵循 Ruff 风格（`ruff format`）
- 通过 Ruff 检查（`ruff check`）
- 使用类型注解
- 编写 Google 风格的 docstring

**示例**:

```python
def process_point_cloud(pc: PointCloud, voxel_size: float) -> PointCloud:
    """处理点云进行体素下采样。

    Args:
        pc: 输入点云
        voxel_size: 体素大小（米）

    Returns:
        下采样后的点云

    Raises:
        ValueError: 当 voxel_size <= 0 时
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    return pc.voxel_downsample(voxel_size)
```

### 测试

- 为新功能添加测试
- 测试覆盖主要代码路径
- 使用 pytest 编写 Python 测试
- 使用 `#[cfg(test)]` 编写 Rust 测试

## 文档

### 更新文档

当您添加新功能或修改 API 时，请更新文档：

1. **API 文档**: 在 `docs/api/` 中更新对应的 markdown 文件
2. **使用指南**: 在 `docs/getting-started/` 中更新
3. **示例**: 在 `docs/getting-started/examples.md` 中添加示例

### 本地预览文档

```bash
just docs-serve
```

访问 http://127.0.0.1:8000 查看效果。

## Pull Request 流程

1. **PR 标题**: 使用清晰的标题
2. **描述**: 详细说明更改内容
3. **关联 Issue**: 使用 `Fixes #123` 链接相关 Issue
4. **检查列表**: 确保所有检查项都完成

**PR 模板**:

```markdown
## 更改内容

简要描述此 PR 的更改

## 相关 Issue

Fixes #123

## 类型

- [ ] Bug 修复
- [ ] 新功能
- [ ] 性能改进
- [ ] 文档更新
- [ ] 代码重构

## 测试

- [ ] 添加了新测试
- [ ] 所有测试通过
- [ ] 手动测试通过

## 检查列表

- [ ] 代码符合项目规范
- [ ] 通过 `just fmt` 格式化
- [ ] 通过 `just lint` 检查
- [ ] 添加/更新了文档
```

## Review 流程

1. **自动检查**: CI 会自动运行测试和检查
2. **代码 Review**: 维护者会 Review 您的代码
3. **修改**: 根据反馈进行修改
4. **合并**: Review 通过后会合并到 main 分支

## 开发技巧

### 快速迭代

```bash
# 监视文件变化，自动重新构建
cargo watch -x 'run --example my_example'
```

### 调试

```bash
# 启用详细日志
RUST_LOG=debug just test

# 使用 Python 调试器
python -m pdb tests/test_xxx.py
```

## 获得帮助

如果您遇到问题或有疑问：

1. 查看[文档](https://YOUR_USERNAME.github.io/pcl-rustic)
2. 搜索[现有 Issues](https://github.com/YOUR_USERNAME/pcl-rustic/issues)
3. 在[Discussions](https://github.com/YOUR_USERNAME/pcl-rustic/discussions)中提问
4. 联系维护者：liuzhen19@xiaomi.com

## 相关资源

- [开发环境设置](setup.md)
- [API 文档](../api/overview.md)
- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3 指南](https://pyo3.rs/)
