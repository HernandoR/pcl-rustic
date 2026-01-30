/// Parquet格式读写/删，使用 polars 统一表格处理
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::io::table::TableColumns;
use crate::utils::error::Result;
use std::fs;

/// 从Parquet文件读取
pub fn from_parquet(path: &str) -> Result<HighPerformancePointCloud> {
    if !fs::metadata(path).is_ok() {
        return Err(format!("文件不存在: {}", path).into());
    }
    let columns = TableColumns::default();
    HighPerformancePointCloud::from_table_parquet(path, columns)
}

/// 写入Parquet文件
pub fn to_parquet(pc: &HighPerformancePointCloud, path: &str) -> Result<()> {
    if pc.point_count() == 0 {
        return Err("点云为空".into());
    }

    // 验证路径可写
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| e.into())?;
        }
    }

    let columns = TableColumns::default();
    pc.to_table_parquet(path, columns)?;

    Ok(())
}
