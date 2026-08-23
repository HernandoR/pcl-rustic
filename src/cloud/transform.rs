//! Matrix / rigid coordinate transforms (ADR-0002, ADR-0003).
//!
//! CPU-resident clouds run a specialised `R * p + t` kernel over rayon, in
//! f64. This deliberately avoids a general matmul: for the degenerate
//! `[N,3] x [3,3]` shape a tiled matmul kernel wins nothing, and burn's is
//! 3.5x slower than a plain loop (measured; see RFC-0001). The
//! rotation-plus-add form also moves 3 components per point rather than the
//! 4 a homogeneous 4x4 would, and this kernel is memory-bound, so it is the
//! cheaper formulation despite 4-wide vectors suiting SIMD better.
//!
//! Device-resident clouds keep the tensor matmul: rotation applies to the
//! f32 relative coordinates on-device, and the translation folds into the
//! f64 offset only, so the promised f8 precision never depends on GPU float
//! precision.

use crate::backend::B;
use crate::cloud::{Coords, PointCloud};
use crate::error::{Error, Result};
use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;

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

    /// Applies a rotation (3x3, row-major) and translation (length 3).
    /// Dims are carried over unchanged.
    pub fn rigid_transform(
        &self,
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
    ) -> Result<Self> {
        let Some(coords) = self.coords.as_ref() else {
            return Ok(self.clone());
        };
        let mut result = self.clone_without_coords();

        // The offset moves with the cloud in both representations: it is the
        // device anchor and the LAS header origin.
        let mut new_offset = [0.0; 3];
        for (i, row) in rotation.iter().enumerate() {
            new_offset[i] = translation[i]
                + row[0] * self.offset[0]
                + row[1] * self.offset[1]
                + row[2] * self.offset[2];
        }

        result.coords = Some(match coords {
            Coords::Cpu(points) => {
                let mut out = vec![0.0f64; points.len()];
                out.par_chunks_mut(3)
                    .zip(points.par_chunks(3))
                    .for_each(|(o, p)| {
                        for i in 0..3 {
                            o[i] = rotation[i][0] * p[0]
                                + rotation[i][1] * p[1]
                                + rotation[i][2] * p[2]
                                + translation[i];
                        }
                    });
                Coords::Cpu(out)
            }
            Coords::Device(tensor) => {
                // Row-major R^T as f32: element (i, j) of R^T is rotation[j][i].
                let mut r_t = [0f32; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        r_t[i * 3 + j] = rotation[j][i] as f32;
                    }
                }
                let r_t_tensor =
                    Tensor::<B, 2>::from_data(TensorData::new(r_t.to_vec(), [3, 3]), &self.device);
                Coords::Device(Box::new(tensor.clone().matmul(r_t_tensor)))
            }
        });

        result.offset = new_offset;
        Ok(result)
    }
}
