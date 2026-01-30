/// burn张量工具函数：类型转换、维度检查等
use burn::tensor::{Tensor, TensorData};
use burn_wgpu::{Wgpu, WgpuDevice};
use crate::utils::error::{PointCloudError, Result};

pub type Backend = Wgpu<f32, i32>;
pub type Tensor1 = Tensor<Backend, 1>;
pub type Tensor2 = Tensor<Backend, 2>;
pub type Device = WgpuDevice;

pub fn default_device() -> Device {
    WgpuDevice::default()
}

pub fn empty_xyz() -> Tensor2 {
    Tensor::<Backend, 2>::zeros([0, 3], &default_device())
}

/// 检查XYZ张量维度
pub fn validate_xyz_shape(xyz: &[Vec<f32>]) -> Result<()> {
    if xyz.is_empty() {
        return Err(PointCloudError::TensorShapeError("XYZ数据为空".to_string()));
    }

    if !xyz.iter().all(|row| row.len() == 3) {
        return Err(PointCloudError::TensorShapeError("XYZ必须为[M,3]的形状".to_string()));
    }

    Ok(())
}

/// 检查RGB张量维度
pub fn validate_rgb_shape(rgb: &[Vec<u8>]) -> Result<usize> {
    if rgb.is_empty() {
        return Err(PointCloudError::TensorShapeError("RGB数据为空".to_string()));
    }

    if !rgb.iter().all(|row| row.len() == 3) {
        return Err(PointCloudError::TensorShapeError("RGB必须为[M,3]的形状".to_string()));
    }

    Ok(rgb.len())
}

/// 检查intensity张量维度
pub fn validate_intensity_shape(
    intensity: &[f32],
    point_count: usize,
) -> Result<()> {
    if intensity.len() != point_count {
        return Err(PointCloudError::DimensionMismatch {
            expected: point_count,
            actual: intensity.len(),
        });
    }
    Ok(())
}

/// 检查属性张量维度
pub fn validate_attribute_shape(
    attribute: &[f32],
    point_count: usize,
) -> Result<()> {
    if attribute.len() != point_count {
        return Err(PointCloudError::DimensionMismatch {
            expected: point_count,
            actual: attribute.len(),
        });
    }
    Ok(())
}

/// 检查矩阵维度（3x3或4x4）
pub fn validate_matrix_shape(matrix: &[Vec<f32>]) -> Result<(usize, usize)> {
    if matrix.is_empty() {
        return Err(PointCloudError::MatrixError("矩阵为空".to_string()));
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    // 检查所有行的列数一致
    if !matrix.iter().all(|row| row.len() == cols) {
        return Err(PointCloudError::MatrixError("矩阵行列数不一致".to_string()));
    }

    // 只允许3x3或4x4矩阵
    if (rows == 3 && cols == 3) || (rows == 4 && cols == 4) {
        Ok((rows, cols))
    } else {
        Err(PointCloudError::MatrixError(format!(
            "仅支持3x3或4x4矩阵，实际{}x{}",
            rows, cols
        )))
    }
}

/// Vec<Vec<f32>> -> Tensor<[M,N]>
pub fn vec2_to_tensor(data: Vec<Vec<f32>>) -> Result<Tensor2> {
    if data.is_empty() {
        return Err(PointCloudError::TensorShapeError("数据为空".to_string()));
    }

    let rows = data.len();
    let cols = data[0].len();
    if !data.iter().all(|row| row.len() == cols) {
        return Err(PointCloudError::TensorShapeError("数据行列数不一致".to_string()));
    }

    let flat: Vec<f32> = data.into_iter().flatten().collect();
    let tensor = Tensor::<Backend, 2>::from_floats(flat, &default_device()).reshape([rows, cols]);
    Ok(tensor)
}

/// XYZ Vec -> Tensor<[M,3]>
pub fn xyz_to_tensor(xyz: Vec<Vec<f32>>) -> Result<Tensor2> {
    validate_xyz_shape(&xyz)?;
    vec2_to_tensor(xyz)
}

pub fn intensity_to_tensor(intensity: Vec<f32>) -> Tensor1 {
    Tensor::<Backend, 1>::from_floats(intensity, &default_device())
}

pub fn tensor1_len(tensor: &Tensor1) -> usize {
    let shape = tensor.shape();
    shape.dims[0]
}

pub fn tensor2_rows(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims[0]
}

pub fn tensor2_cols(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims[1]
}

pub fn tensor1_to_vec(tensor: &Tensor1) -> Vec<f32> {
    let data: TensorData = tensor.to_data();
    data.value
}

pub fn tensor2_to_vec(tensor: &Tensor2) -> Vec<Vec<f32>> {
    let data: TensorData = tensor.to_data();
    let rows = data.shape.dims[0];
    let cols = data.shape.dims[1];
    data.value
        .chunks(cols)
        .take(rows)
        .map(|chunk| chunk.to_vec())
        .collect()
}

pub fn matrix_to_tensor(matrix: Vec<Vec<f32>>) -> Result<Tensor2> {
    let (rows, cols) = validate_matrix_shape(&matrix)?;
    let tensor = vec2_to_tensor(matrix)?;
    let actual_rows = tensor2_rows(&tensor);
    let actual_cols = tensor2_cols(&tensor);
    if actual_rows != rows || actual_cols != cols {
        return Err(PointCloudError::MatrixError("矩阵维度不一致".to_string()));
    }
    Ok(tensor)
}

