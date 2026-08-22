//! Voxel-grid downsampling (ADR-0003): CPU-side grouping by voxel index,
//! then per-voxel representative selection via a plain enum strategy -- no
//! trait objects, replacing the pre-redesign `Box<dyn DownsampleStrategy>`
//! (superseded; see git history).

use crate::backend::B;
use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use burn::tensor::{Tensor, TensorData};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Per-voxel representative-selection strategy. Variants carry their own
/// parameters, so the "pick the middle index and call it random" bug from
/// the old trait-object strategy cannot recur.
#[derive(Clone, Copy, Debug)]
pub enum DownsampleStrategy {
    /// Pick the point closest to the voxel's coordinate centroid.
    NearestToCentroid,
    /// Pick a uniformly random point from the voxel. Deterministic given a
    /// seed; otherwise seeded from OS entropy.
    Random { seed: Option<u64> },
}

impl PointCloud {
    pub fn voxel_downsample(&self, voxel_size: f64, strategy: DownsampleStrategy) -> Result<Self> {
        // NaN-explicit: rejects NaN as well as zero/negative sizes.
        if voxel_size.is_nan() || voxel_size <= 0.0 {
            return Err(Error::value("voxel_size must be positive"));
        }
        let (Some(coords), true) = (self.coords_rel.as_ref(), self.len > 0) else {
            return Ok(self.clone());
        };

        let n = self.len;
        let rel: Vec<f32> = coords
            .to_data()
            .to_vec::<f32>()
            .expect("coordinate tensor is always f32");
        let offset = self.offset;

        // Voxel index per point, offset-anchored: floor((offset+rel)/voxel_size).
        // Computed in parallel: point clouds are typically large enough for
        // this per-point work to be worth spreading across cores.
        let keys: Vec<[i64; 3]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut key = [0i64; 3];
                for (axis, k) in key.iter_mut().enumerate() {
                    let coord = offset[axis] + rel[i * 3 + axis] as f64;
                    *k = (coord / voxel_size).floor() as i64;
                }
                key
            })
            .collect();

        let mut groups: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, key) in keys.into_iter().enumerate() {
            groups.entry(key).or_default().push(i as u32);
        }

        // Iterate voxels in a canonical sorted order so that `Random`'s rng
        // consumption -- and therefore its output -- is deterministic given
        // a seed, independent of HashMap iteration order.
        let mut sorted_keys: Vec<[i64; 3]> = groups.keys().copied().collect();
        sorted_keys.sort_unstable();

        let mut rng = match strategy {
            DownsampleStrategy::Random { seed: Some(seed) } => ChaCha8Rng::seed_from_u64(seed),
            DownsampleStrategy::Random { seed: None } => {
                ChaCha8Rng::seed_from_u64(rand::rng().random())
            }
            DownsampleStrategy::NearestToCentroid => ChaCha8Rng::seed_from_u64(0), // unused
        };

        let mut selected: Vec<u32> = Vec::with_capacity(sorted_keys.len());
        for key in &sorted_keys {
            let indices = &groups[key];
            let pick = match strategy {
                DownsampleStrategy::Random { .. } => indices[rng.random_range(0..indices.len())],
                DownsampleStrategy::NearestToCentroid => nearest_to_centroid(&rel, offset, indices),
            };
            selected.push(pick);
        }
        selected.sort_unstable();

        // Gather: rebuild coords from selected rows on the same device, and
        // gather every dimension column by the same indices.
        let m = selected.len();
        let mut new_rel = Vec::with_capacity(m * 3);
        for &idx in &selected {
            let base = idx as usize * 3;
            new_rel.extend_from_slice(&rel[base..base + 3]);
        }
        let new_tensor = Tensor::<B, 2>::from_data(TensorData::new(new_rel, [m, 3]), &self.device);

        let mut result = self.clone();
        result.coords_rel = Some(new_tensor);
        result.len = m;
        for (_, dim) in result.dims.iter_mut() {
            *dim = dim.gather(&selected);
        }
        Ok(result)
    }
}

fn nearest_to_centroid(rel: &[f32], offset: [f64; 3], indices: &[u32]) -> u32 {
    let mut centroid = [0.0f64; 3];
    for &idx in indices {
        let base = idx as usize * 3;
        for axis in 0..3 {
            centroid[axis] += offset[axis] + rel[base + axis] as f64;
        }
    }
    let count = indices.len() as f64;
    for c in &mut centroid {
        *c /= count;
    }

    *indices
        .iter()
        .min_by(|&&a, &&b| {
            let da = squared_distance(rel, offset, a, centroid);
            let db = squared_distance(rel, offset, b, centroid);
            da.partial_cmp(&db).expect("distances are always finite")
        })
        .expect("a voxel group is never empty")
}

fn squared_distance(rel: &[f32], offset: [f64; 3], idx: u32, centroid: [f64; 3]) -> f64 {
    let base = idx as usize * 3;
    (0..3)
        .map(|axis| {
            let coord = offset[axis] + rel[base + axis] as f64;
            (coord - centroid[axis]).powi(2)
        })
        .sum()
}
