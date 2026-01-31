/// 多格式IO模块入口
pub mod las_laz;
pub mod table;

use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::Result;
use std::path::Path;

impl HighPerformancePointCloud {
    /// 根据扩展名自动加载点云
    /// 支持: .las/.laz/.csv/.parquet/.pq
    pub fn load_from_file(path: &str, columns: Option<table::TableColumns>) -> Result<Self> {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "las" | "laz" => Self::from_las_laz(path),
            "csv" => {
                let cols = columns.unwrap_or_default();
                Self::from_table_csv(path, b',', cols)
            }
            "parquet" | "pq" => {
                let cols = columns.unwrap_or_default();
                Self::from_table_parquet(path, cols)
            }
            _ => Err(format!("不支持的文件格式: {}", ext).into()),
        }
    }

    /// 根据扩展名自动保存点云
    /// 支持: .las/.laz/.csv/.parquet/.pq
    pub fn save_to_file(&self, path: &str, columns: Option<table::TableColumns>) -> Result<()> {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "las" | "laz" => self.to_las(path, ext == "laz"),
            "csv" => {
                let cols = columns.unwrap_or_default();
                self.to_table_csv(path, b',', cols)
            }
            "parquet" | "pq" => {
                let cols = columns.unwrap_or_default();
                self.to_table_parquet(path, cols)
            }
            _ => Err(format!("不支持的文件格式: {}", ext).into()),
        }
    }
}
