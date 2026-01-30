/// 坐标变换：矩阵乘法批量实现XYZ空间变换
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::CoordinateTransform;
use crate::utils::{error::Result, tensor};
use crate::utils::tensor::Backend;
use burn::tensor::Tensor;

impl CoordinateTransform for HighPerformancePointCloud {
    fn transform(&self, matrix: Vec<Vec<f32>>) -> Result<Self> {
        let (rows, cols) = tensor::validate_matrix_shape(&matrix)?;

        let matrix_tensor = tensor::matrix_to_tensor(matrix)?;
        let matrix_t = matrix_tensor.clone().transpose();

        let mut result = self.clone();
        let xyz = self.xyz_ref().clone();

        match (rows, cols) {
            (3, 3) => {
                // 3x3变换矩阵（旋转/缩放）
                let new_xyz = xyz.matmul(matrix_t);
                *result.xyz_mut() = new_xyz;
            }
            (4, 4) => {
                // 4x4变换矩阵（齐次坐标，旋转+平移）
                let xyz_vec = tensor::tensor2_to_vec(&xyz);
                let mut homo_vec = Vec::with_capacity(xyz_vec.len());
                for point in xyz_vec {
                    homo_vec.push(vec![point[0], point[1], point[2], 1.0]);
                }
                let homo_tensor = tensor::vec2_to_tensor(homo_vec)?;
                let new_homo = homo_tensor.matmul(matrix_t);
                let new_homo_vec = tensor::tensor2_to_vec(&new_homo);

                let mut new_xyz_vec = Vec::with_capacity(new_homo_vec.len());
                for row in new_homo_vec {
                    let w = row[3];
                    new_xyz_vec.push(vec![row[0] / w, row[1] / w, row[2] / w]);
                }
                let new_xyz = tensor::xyz_to_tensor(new_xyz_vec)?;
                *result.xyz_mut() = new_xyz;
            }
            _ => {
                return Err("矩阵维度不支持".into());
            }
        }

        Ok(result)
    }

    fn rigid_transform(
        &self,
        rotation: Vec<Vec<f32>>,
        translation: Vec<f32>,
    ) -> Result<Self> {
        if translation.len() != 3 {
            return Err("平移向量必须为3维".into());
        }

        let (rows, cols) = tensor::validate_matrix_shape(&rotation)?;
        if rows != 3 || cols != 3 {
            return Err("旋转矩阵必须为3x3".into());
        }

        let rotation_tensor = tensor::matrix_to_tensor(rotation)?;
        let rotation_t = rotation_tensor.transpose();
        let translation_tensor = Tensor::<Backend, 2>::from_floats(translation, &tensor::default_device()).reshape([1, 3]);

        let mut result = self.clone();

        let xyz = self.xyz_ref().clone();
        let rotated = xyz.matmul(rotation_t);
        let translated = rotated + translation_tensor;
        *result.xyz_mut() = translated;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::PointCloudCore;

    #[test]
    fn test_3x3_transform() {
        let xyz = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let pc = HighPerformancePointCloud::from_xyz(xyz).unwrap();

        // 缩放矩阵（2x缩放）
        let matrix = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ];

        let result = pc.transform(matrix).unwrap();
        let xyz_result = result.get_xyz();

        assert!(
            (xyz_result[0][0] - 2.0).abs() < 1e-5,
            "X坐标变换失败"
        );
    }

    #[test]
    fn test_rigid_transform() {
        let xyz = vec![vec![1.0, 0.0, 0.0]];
        let pc = HighPerformancePointCloud::from_xyz(xyz).unwrap();

        // 恒等旋转
        let rotation = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let translation = vec![1.0, 2.0, 3.0];

        let result = pc.rigid_transform(rotation, translation).unwrap();
        let xyz_result = result.get_xyz();

        assert!((xyz_result[0][0] - 2.0).abs() < 1e-5);
        assert!((xyz_result[0][1] - 2.0).abs() < 1e-5);
        assert!((xyz_result[0][2] - 3.0).abs() < 1e-5);
    }
}
