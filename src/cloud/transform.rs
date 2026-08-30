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
//!
//! CPU-resident f32 clouds (the default dtype) use the same decomposition as
//! the device path -- rotate the relative coordinates in f32, fold the
//! translation into the f64 offset -- so a given cloud transforms to the
//! same values wherever it lives: `R*(offset+rel) + t = (R*offset + t) +
//! R*rel`.

use crate::backend::B;
use crate::cloud::{Coords, PointCloud};
use crate::error::{Error, Result};
use burn::tensor::Tensor;
use rayon::prelude::*;
use std::sync::Arc;

/// Splits a `side * side` row-major matrix into its linear part and
/// translation. A 4x4 matrix's top-left 3x3 block is the linear part; its
/// top-right column is the translation.
fn decompose(matrix: &[f64], side: usize) -> Result<([[f64; 3]; 3], [f64; 3])> {
    match side {
        3 | 4 => {
            let mut linear = [[0.0; 3]; 3];
            for (i, row) in linear.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = matrix[i * side + j];
                }
            }
            let translation = if side == 4 {
                [matrix[3], matrix[7], matrix[11]]
            } else {
                [0.0; 3]
            };
            Ok((linear, translation))
        }
        other => Err(Error::value(format!(
            "transform matrix must be 3x3 or 4x4, got {other}x{other}"
        ))),
    }
}

impl PointCloud {
    /// Applies a 3x3 linear or 4x4 affine transform, given as `side * side`
    /// row-major f8 values, returning a new cloud. Dims are carried over
    /// unchanged (cloned).
    pub fn transform(&self, matrix: &[f64], side: usize) -> Result<Self> {
        let (linear, translation) = decompose(matrix, side)?;
        self.rigid_transform(&linear, &translation)
    }

    /// As [`Self::transform`], but mutates this cloud: coordinates are
    /// rewritten in their existing buffer and dims are untouched -- no
    /// attribute-column clone and no fresh coordinate allocation.
    pub fn transform_inplace(&mut self, matrix: &[f64], side: usize) -> Result<()> {
        let (linear, translation) = decompose(matrix, side)?;
        self.rigid_transform_inplace(&linear, &translation)
    }

    /// The transformed offset: it moves with the cloud in every
    /// representation, being both the device anchor and the LAS header
    /// origin.
    fn transformed_offset(&self, rotation: &[[f64; 3]; 3], translation: &[f64; 3]) -> [f64; 3] {
        let mut new_offset = [0.0; 3];
        for (i, row) in rotation.iter().enumerate() {
            new_offset[i] = translation[i]
                + row[0] * self.offset[0]
                + row[1] * self.offset[1]
                + row[2] * self.offset[2];
        }
        new_offset
    }

    /// Applies a rotation (3x3, row-major) and translation (length 3),
    /// returning a new cloud. Dims are carried over unchanged (cloned).
    pub fn rigid_transform(
        &self,
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
    ) -> Result<Self> {
        let Some(coords) = self.coords.as_ref() else {
            return Ok(self.clone());
        };
        let mut result = self.clone_without_coords();
        let new_offset = self.transformed_offset(rotation, translation);

        result.coords = Some(match coords {
            Coords::CpuF64(points) => {
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
                Coords::CpuF64(Arc::new(out))
            }
            Coords::CpuF32(rel) => {
                // f32 arithmetic on relative coordinates, matching the
                // device matmul's precision; translation lives entirely in
                // the offset update below.
                let r: [[f32; 3]; 3] =
                    std::array::from_fn(|i| std::array::from_fn(|j| rotation[i][j] as f32));
                let mut out = vec![0.0f32; rel.len()];
                out.par_chunks_mut(3)
                    .zip(rel.par_chunks(3))
                    .for_each(|(o, p)| {
                        for i in 0..3 {
                            o[i] = r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2];
                        }
                    });
                Coords::CpuF32(out)
            }
            Coords::Device(tensor) => {
                // `TensorData: From<[[E; B]; A]>` takes the f8 rotation as-is
                // at shape [3, 3]; `from_data` narrows it to the backend's
                // default float dtype (f32, matching the coordinate tensor),
                // and `transpose` supplies the R^T that right-multiplying
                // `[N, 3]` rows needs. No hand-reordered literal, so a
                // transposition typo is not expressible.
                let r_t = Tensor::<B, 2>::from_data(*rotation, &self.device).transpose();
                Coords::Device(Box::new(tensor.clone().matmul(r_t)))
            }
        });

        result.offset = new_offset;
        Ok(result)
    }

    /// As [`Self::rigid_transform`], but mutates this cloud in place: CPU
    /// representations rewrite their existing buffer (no allocation), the
    /// device representation swaps its tensor handle, and dims are not
    /// cloned. The math is identical to the copying path arm for arm.
    pub fn rigid_transform_inplace(
        &mut self,
        rotation: &[[f64; 3]; 3],
        translation: &[f64; 3],
    ) -> Result<()> {
        let new_offset = self.transformed_offset(rotation, translation);
        let device = self.device.clone();
        let Some(coords) = self.coords.as_mut() else {
            return Ok(());
        };

        match coords {
            Coords::CpuF64(points) => {
                // COW: clones the buffer first if a zero-copy view (or a
                // cloned cloud) still shares it, keeping those views stable
                // snapshots; mutates in place otherwise.
                Arc::make_mut(points).par_chunks_mut(3).for_each(|p| {
                    let (x, y, z) = (p[0], p[1], p[2]);
                    for i in 0..3 {
                        p[i] = rotation[i][0] * x
                            + rotation[i][1] * y
                            + rotation[i][2] * z
                            + translation[i];
                    }
                });
            }
            Coords::CpuF32(rel) => {
                let r: [[f32; 3]; 3] =
                    std::array::from_fn(|i| std::array::from_fn(|j| rotation[i][j] as f32));
                rel.par_chunks_mut(3).for_each(|p| {
                    let (x, y, z) = (p[0], p[1], p[2]);
                    for i in 0..3 {
                        p[i] = r[i][0] * x + r[i][1] * y + r[i][2] * z;
                    }
                });
            }
            Coords::Device(tensor) => {
                // burn tensors are immutable handles, so "in place" here
                // means swapping the handle; the win is skipping the dims
                // clone and the result-cloud construction, not the kernel.
                let r_t = Tensor::<B, 2>::from_data(*rotation, &device).transpose();
                **tensor = tensor.clone().matmul(r_t);
            }
        }

        self.offset = new_offset;
        Ok(())
    }
}
