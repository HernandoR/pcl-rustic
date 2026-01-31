/// 点云核心Trait：定义点云基础能力
/// 包括创建、销毁、批量获取XYZ/intensity/RGB等操作
use crate::utils::error::Result;

/// 点云核心能力Trait
pub trait PointCloudCore {
    /// 获取点云的XYZ坐标（形状[M,3]）
    fn get_xyz(&self) -> Vec<Vec<f32>>;

    /// 获取点数
    fn point_count(&self) -> usize;

    /// 检查是否有intensity属性
    fn has_intensity(&self) -> bool;

    /// 检查是否有RGB属性
    fn has_rgb(&self) -> bool;

    /// 获取intensity数据（如果存在）
    fn get_intensity(&self) -> Option<Vec<f32>>;

    /// 获取RGB数据（3个独立通道）
    fn get_rgb(&self) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)>;

    /// 获取自定义属性名列表
    fn attribute_names(&self) -> Vec<String>;

    /// 获取自定义属性数据
    fn get_attribute(&self, name: &str) -> Option<Vec<f32>>;
}

/// 属性管理Trait：定义属性批量操作能力
pub trait PointCloudProperties {
    /// 设置intensity（覆盖式）
    fn set_intensity(&mut self, intensity: Vec<f32>) -> Result<()>;

    /// 设置RGB（覆盖式，3个独立通道）
    fn set_rgb(&mut self, r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> Result<()>;

    /// 添加自定义属性（重复时报错）
    fn add_attribute(&mut self, name: String, data: Vec<f32>) -> Result<()>;

    /// 设置自定义属性（重复时覆盖）
    fn set_attribute(&mut self, name: String, data: Vec<f32>) -> Result<()>;

    /// 删除自定义属性
    fn remove_attribute(&mut self, name: &str) -> Result<()>;
}
