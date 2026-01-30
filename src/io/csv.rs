/// CSV格式读写/删，使用 polars 统一表格处理
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::Result;
use crate::io::table::{TableColumns};

/// 从CSV文件读取（默认列名）
pub fn from_csv(path: &str, delimiter: u8) -> Result<HighPerformancePointCloud> {
    let columns = TableColumns::default();
    HighPerformancePointCloud::from_table_csv(path, delimiter, columns)
}

/// 写入CSV文件（默认列名）
pub fn to_csv(pc: &HighPerformancePointCloud, path: &str, delimiter: u8) -> Result<()> {
    let columns = TableColumns::default();
    pc.to_table_csv(path, delimiter, columns)
}
