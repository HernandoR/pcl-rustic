/// IO通用Trait：定义多格式读写删统一接口
use crate::utils::error::Result;

pub trait IOConvert {
    /// 从LAS/LAZ文件读取
    fn from_las_laz(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// 从Parquet文件读取
    fn from_parquet(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// 从CSV文件读取
    fn from_csv(path: &str, delimiter: u8) -> Result<Self>
    where
        Self: Sized;

    /// 写入LAS/LAZ文件
    fn to_las(&self, path: &str, compress: bool) -> Result<()>;

    /// 写入Parquet文件
    fn to_parquet(&self, path: &str) -> Result<()>;

    /// 写入CSV文件
    fn to_csv(&self, path: &str, delimiter: u8) -> Result<()>;

    /// 删除指定格式的文件
    fn delete_file(path: &str) -> Result<()>;
}
