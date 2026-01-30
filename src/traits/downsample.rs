/// 下采样通用Trait：定义体素下采样和采样策略规范
use crate::utils::error::Result;

/// 采样策略Trait：定义采样器如何选择代表性点
pub trait DownsampleStrategy: Send + Sync {
    /// 从一个体素内的点中选择代表性点
    /// indices: 体素内点的索引
    /// xyz: 全部点云的XYZ坐标
    /// 返回选中点的索引
    fn select_representative(&self, indices: Vec<usize>, xyz: &[Vec<f32>]) -> Result<usize>;

    /// 返回策略名称
    fn name(&self) -> &str;
}

/// 体素下采样Trait：定义下采样核心接口
pub trait VoxelDownsample {
    /// 执行体素下采样
    /// voxel_size: 体素大小
    /// strategy: 采样策略
    fn voxel_downsample(
        &self,
        voxel_size: f32,
        strategy: Box<dyn DownsampleStrategy>,
    ) -> Result<Self>
    where
        Self: Sized;
}
