/// 坐标变换Trait：定义变换接口
use crate::utils::error::Result;

pub trait CoordinateTransform {
    /// 执行坐标变换（3x3或4x4矩阵）
    /// matrix: 变换矩阵（按行优先顺序存储）
    fn transform(&self, matrix: Vec<Vec<f32>>) -> Result<Self>
    where
        Self: Sized;

    /// 执行刚体变换（旋转+平移）
    /// rotation: 3x3旋转矩阵
    /// translation: 3维平移向量
    fn rigid_transform(&self, rotation: Vec<Vec<f32>>, translation: Vec<f32>) -> Result<Self>
    where
        Self: Sized;
}
