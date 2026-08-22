//! `PointCloud`: the single mutation choke point for all dimension data
//! (ADR-0002, ADR-0003). Coordinates ride a tensor on the compute device as
//! f32-relative values anchored to an f64 offset; every other dimension is a
//! CPU-side `DimArray` column, ordered by first insertion.
//!
//! `transform.rs` and `voxel.rs` are submodules of `cloud` and read/write
//! `PointCloud`'s private fields directly (Rust module privacy allows this
//! for descendant modules); `io/` and `py/` only ever go through the public
//! methods defined here, which enforce the length invariant.

pub mod dims;
pub mod transform;
pub mod voxel;

use crate::backend::{self, DispatchDevice, B};
use crate::cloud::dims::{is_coordinate_name, standard_dim_dtype, DType, DimArray};
use crate::error::{Error, Result};
use burn::tensor::{Tensor, TensorData};

/// A point cloud: named, equal-length dimensions plus device placement.
#[derive(Clone)]
pub struct PointCloud {
    /// Coordinates relative to `offset`, f32, shape `[len, 3]`. `None` until
    /// the first coordinate write, so a freshly constructed cloud never
    /// allocates a device tensor.
    coords_rel: Option<Tensor<B, 2>>,
    /// Per-cloud f64 anchor added back on every coordinate read (ADR-0002).
    offset: [f64; 3],
    /// LAS scale factors, kept only for LAS write fidelity (ADR-0002).
    scales: [f64; 3],
    /// Non-coordinate dimensions, ordered by first insertion.
    dims: Vec<(String, DimArray)>,
    /// Common length shared by every established dimension.
    len: usize,
    device: DispatchDevice,
}

impl PointCloud {
    pub fn new() -> Self {
        Self {
            coords_rel: None,
            offset: [0.0; 3],
            scales: [0.001; 3],
            dims: Vec::new(),
            len: 0,
            device: backend::default_device(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn device(&self) -> &DispatchDevice {
        &self.device
    }

    fn is_established(&self) -> bool {
        self.coords_rel.is_some() || !self.dims.is_empty()
    }

    /// Single mutation choke point: every write that changes (or first
    /// establishes) the cloud's point count goes through here.
    fn check_or_set_len(&mut self, n: usize) -> Result<()> {
        if self.is_established() {
            if n != self.len {
                return Err(Error::value(format!(
                    "dimension length {n} does not match the point cloud's length {}",
                    self.len
                )));
            }
        } else {
            self.len = n;
        }
        Ok(())
    }

    fn dim_index(&self, name: &str) -> Option<usize> {
        self.dims.iter().position(|(n, _)| n == name)
    }

    // ---- coordinate plane ---------------------------------------------

    pub fn offset(&self) -> [f64; 3] {
        self.offset
    }

    pub fn scales(&self) -> [f64; 3] {
        self.scales
    }

    pub fn set_scales(&mut self, scales: [f64; 3]) {
        self.scales = scales;
    }

    /// Row-major `[x0,y0,z0,x1,y1,z1,...]` f8 coordinates, or `None` if
    /// coordinates were never established.
    pub fn coords_f64(&self) -> Option<Vec<f64>> {
        let tensor = self.coords_rel.as_ref()?;
        let rel: Vec<f32> = tensor
            .to_data()
            .to_vec::<f32>()
            .expect("coordinate tensor is always f32");
        let mut out = Vec::with_capacity(rel.len());
        for chunk in rel.as_chunks::<3>().0 {
            out.push(self.offset[0] + chunk[0] as f64);
            out.push(self.offset[1] + chunk[1] as f64);
            out.push(self.offset[2] + chunk[2] as f64);
        }
        Some(out)
    }

    /// A single coordinate axis (`0=x, 1=y, 2=z`) as f8, or `None` if
    /// coordinates were never established.
    pub fn axis_f64(&self, axis: usize) -> Option<Vec<f64>> {
        let tensor = self.coords_rel.as_ref()?;
        let rel: Vec<f32> = tensor
            .to_data()
            .to_vec::<f32>()
            .expect("coordinate tensor is always f32");
        Some(
            rel.as_chunks::<3>()
                .0
                .iter()
                .map(|c| self.offset[axis] + c[axis] as f64)
                .collect(),
        )
    }

    /// Establishes or replaces coordinates wholesale from row-major f8 data,
    /// re-anchoring `offset` to the first point (ADR-0002 "ingest anchors
    /// offset at first point"). Used by `from_xyz`, `set_dim("xyz", ...)`,
    /// and CSV/Parquet ingest.
    pub fn set_coords_f64(&mut self, xyz: &[f64]) -> Result<()> {
        self.set_coords_f64_impl(xyz, None)
    }

    /// As `set_coords_f64`, but anchors `offset` explicitly instead of to
    /// the first point. Used by LAS ingest, which anchors to the header's
    /// own scale/offset (ADR-0002).
    pub(crate) fn set_coords_f64_with_offset(
        &mut self,
        xyz: &[f64],
        offset: [f64; 3],
    ) -> Result<()> {
        self.set_coords_f64_impl(xyz, Some(offset))
    }

    fn set_coords_f64_impl(&mut self, xyz: &[f64], offset: Option<[f64; 3]>) -> Result<()> {
        if !xyz.len().is_multiple_of(3) {
            return Err(Error::value("xyz data length must be a multiple of 3"));
        }
        let n = xyz.len() / 3;
        self.check_or_set_len(n)?;
        let offset = offset.unwrap_or_else(|| {
            if n > 0 {
                [xyz[0], xyz[1], xyz[2]]
            } else {
                [0.0; 3]
            }
        });
        let rel: Vec<f32> = xyz
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|c| {
                [
                    (c[0] - offset[0]) as f32,
                    (c[1] - offset[1]) as f32,
                    (c[2] - offset[2]) as f32,
                ]
            })
            .collect();
        self.offset = offset;
        self.coords_rel = Some(Tensor::<B, 2>::from_data(
            TensorData::new(rel, [n, 3]),
            &self.device,
        ));
        Ok(())
    }

    /// Sets one coordinate axis (`0=x, 1=y, 2=z`) in place, using the
    /// existing offset. If this is the cloud's first coordinate write, the
    /// other two axes are established at zero (offset stays `[0,0,0]`).
    pub fn set_axis_f64(&mut self, axis: usize, values: &[f64]) -> Result<()> {
        let n = values.len();
        self.check_or_set_len(n)?;
        let mut rel = match &self.coords_rel {
            Some(t) => t
                .to_data()
                .to_vec::<f32>()
                .expect("coordinate tensor is always f32"),
            None => vec![0.0f32; n * 3],
        };
        for (row, &v) in values.iter().enumerate() {
            rel[row * 3 + axis] = (v - self.offset[axis]) as f32;
        }
        self.coords_rel = Some(Tensor::<B, 2>::from_data(
            TensorData::new(rel, [n, 3]),
            &self.device,
        ));
        Ok(())
    }

    // ---- dimension plane ------------------------------------------------

    pub fn has_dim(&self, name: &str) -> bool {
        if is_coordinate_name(name) {
            return self.coords_rel.is_some();
        }
        self.dim_index(name).is_some()
    }

    pub fn dim_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.dims.len() + 3);
        if self.coords_rel.is_some() {
            names.push("x".to_string());
            names.push("y".to_string());
            names.push("z".to_string());
        }
        names.extend(self.dims.iter().map(|(n, _)| n.clone()));
        names
    }

    pub fn dim_dtypes(&self) -> Vec<(String, DType)> {
        let mut out = Vec::with_capacity(self.dims.len() + 3);
        if self.coords_rel.is_some() {
            out.push(("x".to_string(), DType::F64));
            out.push(("y".to_string(), DType::F64));
            out.push(("z".to_string(), DType::F64));
        }
        out.extend(self.dims.iter().map(|(n, d)| (n.clone(), d.dtype())));
        out
    }

    pub fn dim(&self, name: &str) -> Option<&DimArray> {
        self.dim_index(name).map(|i| &self.dims[i].1)
    }

    /// The dtype `set_dim`/`add_extra_dim` must observe for `name`: the
    /// pinned standard dtype, or the previously declared extra-dim dtype.
    /// `None` for an undeclared, non-standard name.
    pub fn required_dim_dtype(&self, name: &str) -> Option<DType> {
        standard_dim_dtype(name).or_else(|| self.dim(name).map(|d| d.dtype()))
    }

    /// Sets (or replaces) a non-coordinate dimension. File-format readers
    /// use this to define columns on the fly; the stricter Python-facing
    /// `set_dim` requires `add_extra_dim` first for non-standard names.
    pub fn set_dim_array(&mut self, name: &str, data: DimArray) -> Result<()> {
        if is_coordinate_name(name) {
            return Err(Error::value(format!(
                "'{name}' is a coordinate dimension; set it via the xyz/x/y/z coordinate API"
            )));
        }
        self.check_or_set_len(data.len())?;
        match self.dim_index(name) {
            Some(i) => self.dims[i].1 = data,
            None => self.dims.push((name.to_string(), data)),
        }
        Ok(())
    }

    pub fn add_extra_dim(&mut self, name: &str, dtype: DType) -> Result<()> {
        if is_coordinate_name(name) || standard_dim_dtype(name).is_some() {
            return Err(Error::value(format!(
                "'{name}' is a standard dimension name and cannot be declared as an extra dimension"
            )));
        }
        if self.dim_index(name).is_some() {
            return Err(Error::value(format!("dimension '{name}' already exists")));
        }
        self.dims
            .push((name.to_string(), DimArray::zeros(self.len, dtype)));
        Ok(())
    }

    pub fn remove_dim(&mut self, name: &str) -> Result<()> {
        if is_coordinate_name(name) {
            return Err(Error::value(format!(
                "'{name}' is a coordinate dimension and cannot be removed"
            )));
        }
        match self.dim_index(name) {
            Some(i) => {
                self.dims.remove(i);
                Ok(())
            }
            None => Err(Error::key(format!("no such dimension '{name}'"))),
        }
    }

    // ---- device -----------------------------------------------------------

    pub fn to_device(&self, device: DispatchDevice) -> Self {
        let mut cloned = self.clone();
        if let Some(t) = cloned.coords_rel.take() {
            cloned.coords_rel = Some(t.to_device(&device));
        }
        cloned.device = device;
        cloned
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}
