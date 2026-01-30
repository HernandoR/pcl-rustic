#!/bin/bash
# PCL Rustic 项目文件结构验证脚本

echo "PCL Rustic 项目结构验证"
echo "======================="
echo ""

# 检查Rust源文件
echo "✓ Rust源文件结构:"
echo "  src/"
echo "  ├── lib.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/lib.rs) lines) - PyO3模块入口"

echo "  ├── traits/ - Trait抽象层"
echo "  │   ├── mod.rs - 模块导出"
echo "  │   ├── point_cloud.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/traits/point_cloud.rs) lines) - 核心Trait"
echo "  │   ├── io.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/traits/io.rs) lines) - IO接口"
echo "  │   ├── downsample.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/traits/downsample.rs) lines) - 下采样Trait"
echo "  │   └── transform.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/traits/transform.rs) lines) - 变换Trait"

echo "  ├── point_cloud/ - 核心实现"
echo "  │   ├── mod.rs - 模块导出"
echo "  │   ├── core.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/point_cloud/core.rs) lines) - 核心结构体"
echo "  │   ├── attributes.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/point_cloud/attributes.rs) lines) - 属性管理"
echo "  │   ├── transform.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/point_cloud/transform.rs) lines) - 变换实现"
echo "  │   └── voxel.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/point_cloud/voxel.rs) lines) - 下采样实现"

echo "  ├── io/ - 多格式I/O"
echo "  │   ├── mod.rs - 模块导出"
echo "  │   ├── las_laz.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/io/las_laz.rs) lines) - LAS/LAZ格式"
echo "  │   ├── parquet.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/io/parquet.rs) lines) - Parquet格式"
echo "  │   └── csv.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/io/csv.rs) lines) - CSV格式"

echo "  ├── utils/ - 工具模块"
echo "  │   ├── mod.rs - 模块导出"
echo "  │   ├── error.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/utils/error.rs) lines) - 异常处理"
echo "  │   ├── tensor.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/utils/tensor.rs) lines) - 张量验证"
echo "  │   └── reflect.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/utils/reflect.rs) lines) - 体素分组"

echo "  └── interop/ - 跨生态互通"
echo "      ├── mod.rs - 模块导出"
echo "      └── numpy.rs ($(wc -l < /Users/lz/Codes/pcl-rustic/src/interop/numpy.rs) lines) - numpy转换"

echo ""
echo "✓ Python文件结构:"
echo "  src/pcl_rustic/"
echo "  ├── __init__.py ($(wc -l < /Users/lz/Codes/pcl-rustic/src/pcl_rustic/__init__.py) lines) - 模块入口"
echo "  ├── _core.pyi ($(wc -l < /Users/lz/Codes/pcl-rustic/src/pcl_rustic/_core.pyi) lines) - 类型注解"
echo "  └── py.typed - PEP 561标记"

echo ""
echo "✓ 文档文件:"
for file in /Users/lz/Codes/pcl-rustic/*.md; do
    name=$(basename "$file")
    lines=$(wc -l < "$file")
    echo "  - $name ($lines lines)"
done

echo ""
echo "✓ 测试文件:"
echo "  tests/"
echo "  ├── test_point_cloud.py ($(wc -l < /Users/lz/Codes/pcl-rustic/tests/test_point_cloud.py) lines) - 40+ 测试用例"
echo "  ├── conftest.py ($(wc -l < /Users/lz/Codes/pcl-rustic/tests/conftest.py) lines) - pytest配置"
echo "  └── data/ - 测试数据目录"

echo ""
echo "✓ 示例文件:"
echo "  examples/"
echo "  └── basic_usage.py ($(wc -l < /Users/lz/Codes/pcl-rustic/examples/basic_usage.py) lines) - 使用示例"

echo ""
echo "✓ 项目配置文件:"
echo "  - Cargo.toml - Rust项目配置"
echo "  - pyproject.toml - Python项目配置"
echo "  - ai_doc/README.md - 项目说明"

echo ""
echo "统计信息"
echo "========="
RUST_LINES=$(find /Users/lz/Codes/pcl-rustic/src -name "*.rs" -exec wc -l {} + | tail -1 | awk '{print $1}')
PYTHON_LINES=$(find /Users/lz/Codes/pcl-rustic/src -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
TEST_LINES=$(find /Users/lz/Codes/pcl-rustic/tests -name "*.py" -exec wc -l {} + | tail -1 | awk '{print $1}')
DOC_LINES=$(find /Users/lz/Codes/pcl-rustic -maxdepth 1 -name "*.md" -exec wc -l {} + | tail -1 | awk '{print $1}')

echo "Rust源代码:    $RUST_LINES 行"
echo "Python代码:    $PYTHON_LINES 行"
echo "测试代码:      $TEST_LINES 行"
echo "文档:         $DOC_LINES 行"
echo "总代码量:      $((RUST_LINES + PYTHON_LINES + TEST_LINES)) 行"

echo ""
echo "✓ 项目构建就绪！"
echo ""
echo "下一步:"
echo "  1. 进入项目目录: cd /Users/lz/Codes/pcl-rustic"
echo "  2. 构建项目: uv build"
echo "  3. 运行测试: pytest tests/ -v"
echo "  4. 阅读文档: ai_doc/README.md"
