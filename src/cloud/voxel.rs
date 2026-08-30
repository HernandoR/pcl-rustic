//! Voxel-grid downsampling (ADR-0003): CPU-side grouping by voxel index,
//! then per-voxel representative selection via a plain enum strategy -- no
//! trait objects, replacing the pre-redesign `Box<dyn DownsampleStrategy>`
//! (superseded; see git history).

use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

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
    /// Group points into voxels and keep one representative per voxel.
    ///
    /// Grouping is sort-based (PCL's approach), not hash-based: a linear
    /// voxel key is computed per point, point indices are sorted once by
    /// that key, and the sorted order is scanned in a single pass where
    /// each run of equal keys is a voxel. This avoids a `HashMap<[i64; 3],
    /// Vec<u32>>` -- one heap allocation per voxel, a separately sorted key
    /// vector, and a second hash lookup per voxel during selection -- which
    /// made grouping cost scale with voxel count rather than point count.
    pub fn voxel_downsample(&self, voxel_size: f64, strategy: DownsampleStrategy) -> Result<Self> {
        // NaN-explicit: rejects NaN as well as zero/negative sizes.
        if voxel_size.is_nan() || voxel_size <= 0.0 {
            return Err(Error::value("voxel_size must be positive"));
        }
        if self.len == 0 {
            return Ok(self.clone());
        }

        // Borrow the absolute f64 coordinates without copying when the
        // cloud is CPU-f64; otherwise materialize once. Either way the
        // offset is already folded in, so voxel keys need no further
        // offset arithmetic.
        let Some(xyz) = self.coords_f64_cow() else {
            return Ok(self.clone());
        };
        let xyz: &[f64] = &xyz;

        let n = self.len;

        // Voxel key per point, offset-anchored: floor(coord / voxel_size).
        // Computed in parallel: point clouds are typically large enough for
        // this per-point work to be worth spreading across cores.
        let keys: Vec<[i64; 3]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let base = i * 3;
                std::array::from_fn(|axis| (xyz[base + axis] / voxel_size).floor() as i64)
            })
            .collect();

        // Sort point indices by voxel key once. No per-voxel allocation:
        // every voxel's members end up as a contiguous slice of `order`.
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.par_sort_unstable_by_key(|&i| keys[i as usize]);

        // Iterate voxels in ascending key order -- the same order the
        // sorted-key-vector approach produced -- so that `Random`'s rng
        // consumption, and therefore its output, stays deterministic given
        // a seed, independent of any hashing.
        let mut rng = match strategy {
            DownsampleStrategy::Random { seed: Some(seed) } => ChaCha8Rng::seed_from_u64(seed),
            DownsampleStrategy::Random { seed: None } => {
                ChaCha8Rng::seed_from_u64(rand::rng().random())
            }
            DownsampleStrategy::NearestToCentroid => ChaCha8Rng::seed_from_u64(0), // unused
        };

        // Single pass over the sorted order: each run of equal keys is one
        // voxel, its members a slice `&order[start..end]` -- no lookup, no
        // allocation beyond the output vector itself.
        let mut selected: Vec<u32> = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let key = keys[order[start] as usize];
            let mut end = start + 1;
            while end < order.len() && keys[order[end] as usize] == key {
                end += 1;
            }
            let run = &order[start..end];
            let pick = match strategy {
                DownsampleStrategy::Random { .. } => run[rng.random_range(0..run.len())],
                DownsampleStrategy::NearestToCentroid => nearest_to_centroid(xyz, run),
            };
            selected.push(pick);
            start = end;
        }
        selected.sort_unstable();

        self.with_gathered_rows(&selected)
    }
}

fn nearest_to_centroid(xyz: &[f64], indices: &[u32]) -> u32 {
    let mut centroid = [0.0f64; 3];
    for &idx in indices {
        let base = idx as usize * 3;
        for axis in 0..3 {
            centroid[axis] += xyz[base + axis];
        }
    }
    let count = indices.len() as f64;
    for c in &mut centroid {
        *c /= count;
    }

    *indices
        .iter()
        .min_by(|&&a, &&b| {
            let da = squared_distance(xyz, a, centroid);
            let db = squared_distance(xyz, b, centroid);
            da.partial_cmp(&db).expect("distances are always finite")
        })
        .expect("a voxel group is never empty")
}

fn squared_distance(xyz: &[f64], idx: u32, centroid: [f64; 3]) -> f64 {
    let base = idx as usize * 3;
    (0..3)
        .map(|axis| {
            let coord = xyz[base + axis];
            (coord - centroid[axis]).powi(2)
        })
        .sum()
}
