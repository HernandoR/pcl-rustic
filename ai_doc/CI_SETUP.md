# CI/CD 配置说明

## 📋 概述

本项目采用分阶段 CI/CD 流程，通过 3 个独立但相互依赖的工作流实现自动化测试和性能基准测试。

## 🔗 工作流依赖链

```
Pre-commit Checks → Test → Benchmark
```

每个阶段可独立手动触发，但自动触发时遵循依赖关系。

## 🔧 工作流文件

### 1. `.github/workflows/pre-commit.yml` - 代码质量检查

**触发条件**：
```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:  # 手动触发
```

**执行内容**：
- ✅ 运行 pre-commit hooks（ruff format, ruff check）
- ✅ Rust 代码格式检查（rustfmt）
- ✅ Rust linter 检查（clippy）

**运行环境**：
- 操作系统：`ubuntu-latest`
- Python 版本：`3.11`

---

### 2. `.github/workflows/test.yml` - 多平台测试

**触发条件**：
```yaml
on:
  push:
    branches: [main]  # 仅 main 分支自动触发
  workflow_dispatch:  # 手动触发
  workflow_run:
    workflows: ["Pre-commit Checks"]
    types: [completed]  # Pre-commit 完成后触发
```

**依赖关系**：
- 只有在 pre-commit 通过或手动触发时才运行

**执行内容**：
- ✅ **测试矩阵**：
  - 操作系统：`ubuntu-latest`, `macos-latest`, `windows-latest`
  - Python 版本：`3.9`, `3.10`, `3.11`, `3.12`
  - Windows 排除组合（3.10, 3.11）以节省时间

- ✅ **测试步骤**：
  1. 构建 Rust 扩展（release 模式）
  2. 运行 Rust 测试
  3. 运行 Python 测试（排除慢速测试）

- ✅ **构建 Wheels**：
  - 依赖 test job 完成
  - 构建 3 个平台的 wheel 包
  - 上传为 artifacts（保留 7 天）

---

### 3. `.github/workflows/benchmark.yml` - 性能基准测试

**触发条件**：
```yaml
on:
  push:
    tags:
      - 'v*.*.*'  # 发布 release tag 时自动触发
  workflow_dispatch:  # 手动触发
  workflow_run:
    workflows: ["Test"]
    types: [completed]  # Test 完成后触发
```

**依赖关系**：
- 只有在 test 通过、release tag 或手动触发时才运行

**执行内容**：
- ✅ **基准测试矩阵**：
  - 操作系统：`ubuntu-latest`, `macos-latest`, `windows-latest`
  - Python 版本：`3.11`

- ✅ **测试步骤**：
  1. 构建 release 版本
  2. 运行完整基准测试
  3. 上传测试结果（保留 30 天）
  4. 在 PR 上评论结果（如果适用）

---

## 📈 自动触发流程

### 场景 1：PR 到 main/develop
```
1. Push 代码到 PR 分支
2. ✅ Pre-commit Checks 自动运行
3. ❌ Test 不会自动运行（只在 main 分支触发）
4. ❌ Benchmark 不会自动运行
```

### 场景 2：Push 到 develop 分支
```
1. Push 代码到 develop
2. ✅ Pre-commit Checks 自动运行
3. ❌ Test 不会自动运行（只在 main 分支触发）
4. ❌ Benchmark 不会自动运行
```

### 场景 3：Push 到 main 分支（合并 PR）
```
1. Push/Merge 代码到 main
2. ✅ Pre-commit Checks 自动运行
3. ✅ Test 自动运行（pre-commit 通过后）
   - 运行完整测试矩阵
   - 构建 wheels
4. ❌ Benchmark 不会自动运行（需要 release tag）
```

### 场景 4：发布 Release Tag
```
1. 创建并 push release tag (v1.0.0)
2. ✅ Benchmark 自动运行
   - 在所有平台运行基准测试
   - 上传结果作为 artifacts
```

---

## 🎯 手动触发

所有 3 个工作流都支持手动触发：

### 在 GitHub UI 手动触发

1. 进入仓库的 **Actions** 页面
2. 选择要运行的工作流：
   - Pre-commit Checks
   - Test
   - Benchmark
3. 点击 **Run workflow**
4. 选择分支
5. 点击绿色的 **Run workflow** 按钮

### 使用 GitHub CLI 手动触发

```bash
# 触发 pre-commit
gh workflow run pre-commit.yml

# 触发 test
gh workflow run test.yml

# 触发 benchmark
gh workflow run benchmark.yml
```

---

## 🎯 测试矩阵

### Pre-commit Checks（1 个任务）

| OS | Python 版本 |
|---------|-----------|
| Ubuntu | 3.11 |

### Test Job 矩阵（10 个组合）

| OS | Python 版本 |
|---------|-----------|
| Ubuntu | 3.9, 3.10, 3.11, 3.12 |
| macOS | 3.9, 3.10, 3.11, 3.12 |
| Windows | 3.9, 3.12 |

### Benchmark Job 矩阵（3 个组合）

| OS | Python 版本 |
|---------|-----------|
| Ubuntu | 3.11 |
| macOS | 3.11 |
| Windows | 3.11 |

---

## 🔍 代码质量工具

### Rust
- **rustfmt**：代码格式化
- **clippy**：静态分析和 linter

### Python
- **ruff**：高性能 linter 和 formatter（已配置在 `.pre-commit-config.yaml`）
- **pre-commit**：Git hook 管理

---

## 📝 使用示例

### 本地开发

```bash
# 1. 安装依赖
uv sync --dev

# 2. 安装 pre-commit hooks
pre-commit install

# 3. 开发构建
maturin develop --release

# 4. 运行测试
uv run pytest tests/ -v

# 5. 运行代码检查
pre-commit run --all-files
```

### 触发 CI

```bash
# 推送到 develop - 只触发 pre-commit
git push origin develop

# 推送到 main - 触发 pre-commit + test
git push origin main

# 发布 release - 触发 benchmark
git tag v1.0.0
git push origin v1.0.0
```

---

## 📊 CI 输出示例

### 测试通过
```
✅ Pre-commit checks - Passed (45s)
✅ Test on ubuntu-latest / Python 3.11 - Passed (2m 34s)
✅ Test on macos-latest / Python 3.11 - Passed (3m 12s)
✅ Test on windows-latest / Python 3.11 - Passed (4m 56s)
✅ Build wheels on ubuntu-latest - Passed (1m 23s)
```

### Benchmark 结果
基准测试结果会上传为 artifact，可在 Actions 页面下载：
- `benchmark-results-ubuntu-latest`
- `benchmark-results-macos-latest`
- `benchmark-results-windows-latest`

---

## 🚀 发布流程

完整的发布流程：

```bash
# 1. 确保在 main 分支
git checkout main
git pull origin main

# 2. 更新版本号
# 编辑 Cargo.toml 和 pyproject.toml

# 3. 提交版本更新
git add Cargo.toml pyproject.toml
git commit -m "chore: bump version to 1.0.0"
git push origin main

# 4. 创建并推送 tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# 5. 等待 CI 完成
# - Pre-commit Checks 会自动运行
# - Test 会自动运行
# - Benchmark 会自动运行（因为是 tag）

# 6. 下载 wheels 从 Actions artifacts
# 7. 上传到 PyPI（手动或通过 release action）
```

---

## 🐛 常见问题

### Q: Pre-commit 失败了怎么办？
A: 本地运行 `pre-commit run --all-files` 修复格式问题，然后重新提交。

### Q: Test 没有在 PR 中自动运行？
A: 正常，test 只在 main 分支自动运行。可以手动触发测试。

### Q: 如何跳过 CI？
A: 在 commit message 中添加 `[skip ci]` 或 `[ci skip]`。

### Q: 如何只运行特定平台的测试？
A: 不支持，需要修改 workflow 文件的矩阵配置。

### Q: Benchmark 可以在 PR 中运行吗？
A: 可以手动触发，但不会自动运行。

---

## 📚 相关文档

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [Matrix Strategy 指南](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstrategymatrix)
- [Workflow Dependencies](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_run)
- [uv 文档](https://docs.astral.sh/uv/)
- [pre-commit 文档](https://pre-commit.com/)
- [ruff 文档](https://docs.astral.sh/ruff/)

---

配置完成日期：2026年1月31日
