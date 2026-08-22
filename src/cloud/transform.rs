//! Matrix / rigid coordinate transforms (ADR-0002, ADR-0003): rotation runs
//! as an f32 tensor matmul on the compute device, translation folds into the
//! f64 offset only, so the promised f8 coordinate precision never depends on
//! GPU float precision.

use crate::backend::B;
use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use burn::tensor::{Tensor, TensorData};

impl PointCloud {
    /// Applies a 3x3 linear or 4x4 affine transform, given as `side * side`
    /// row-major f8 values. A 4x4 matrix's top-left 3x3 block is the linear
    /// part; its top-right column is the translation. Dims are carried over
    /// unchanged.
    pub fn transform(&self, matrix: &[f64], side: usize) -> Result<Self> {
        match side {
            3 => {
                let mut linear = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        linear[i][j] = matrix[i * 3 + j];
                    }
                }
                self.rigid_transform(&linear, &[0.0, 0.0, 0.0])
            }
            4 => {
                let mut linear = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        linear[i][j] = matrix[i * 4 + j];
                    }
                }
                let translation = [matrix[3], matrix[7], matrix[11]];
                self.rigid_transform(&linear, &translation)
            }
            other => Err(Error::value(format!(
                "transform matrix must be 3x3 or 4x4, got {other}x{other}"
            ))),
        }
    }

    /// Applies a rotation (3x3, row-major) and translation (length 3) in one
    /// step: `new_offset = R . offset + t` (exact f64); `rel' = rel . R^T`
    /// (f32 tensor matmul on-device). Dims are carried over unchanged.
    pub fn rigid_transform(
        &self,
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
    ) -> Result<Self> {
        let mut result = self.clone();

        let Some(coords) = self.coords_rel.as_ref() else {
            return Ok(result);
        };

        let mut new_offset = [0.0; 3];
        for (i, row) in rotation.iter().enumerate() {
            new_offset[i] = translation[i]
                + row[0] * self.offset[0]
                + row[1] * self.offset[1]
                + row[2] * self.offset[2];
        }

        // Row-major R^T as f32: element (i, j) of R^T is rotation[j][i].
        let mut r_t = [0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                r_t[i * 3 + j] = rotation[j][i] as f32;
            }
        }
        let r_t_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(r_t.to_vec(), [3, 3]), &self.device);
        let rel_t = coords.clone().matmul(r_t_tensor);

        result.coords_rel = Some(rel_t);
        result.offset = new_offset;
        Ok(result)
    }
}
