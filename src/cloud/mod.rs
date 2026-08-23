//! `PointCloud`: the single mutation choke point for all dimension data
//! (ADR-0002, ADR-0003).
//!
//! Coordinates have two representations, picked by device placement:
//!
//! * **CPU-resident** clouds hold absolute f64 coordinates in a plain `Vec`.
//!   This is the common case. It makes `x`/`y`/`z` exact (no f32 rounding),
//!   and it removes the f32 -> f64 conversion that used to dominate every
//!   coordinate read.
//! * **Device-resident** clouds hold f32 values relative to `offset`, since
//!   wgpu has no f64. Precision then depends on the cloud's extent, which is
//!   why `offset` anchors near the data.
//!
//! `offset` and `scales` are retained in both cases: they are LAS header
//! metadata and the anchor for the device representation.
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

/// How a cloud's coordinates are stored. See the module comment.
#[derive(Clone)]
pub(crate) enum Coords {
    /// Absolute f64, row-major `[x0,y0,z0,...]`, CPU-resident.
    Cpu(Vec<f64>),
    /// f32 relative to the cloud's `offset`, shape `[len, 3]`, on-device.
    /// Boxed: a burn `Tensor` handle is far larger than a `Vec`, and this
    /// variant is the uncommon one.
    Device(Box<Tensor<B, 2>>),
}

/// A point cloud: named, equal-length dimensions plus device placement.
#[derive(Clone)]
pub struct PointCloud {
    /// `None` until the first coordinate write, so a freshly constructed
    /// cloud allocates nothing.
    pub(crate) coords: Option<Coords>,
    /// Per-cloud f64 anchor: the origin of the device representation, and
    /// the LAS header offset (ADR-0002).
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
            coords: None,
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
        self.coords.is_some() || !self.dims.is_empty()
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

    /// Clones everything except the coordinate buffer, for operations that
    /// immediately overwrite it. Cloning `self` wholesale would copy the
    /// entire coordinate plane (240 MB at 10M points) only to drop it.
    pub(crate) fn clone_without_coords(&self) -> Self {
        Self {
            coords: None,
            offset: self.offset,
            scales: self.scales,
            dims: self.dims.clone(),
            len: self.len,
            device: self.device.clone(),
        }
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
    /// coordinates were never established. CPU-resident clouds copy their
    /// buffer; device-resident clouds transfer and widen f32 -> f64.
    pub fn coords_f64(&self) -> Option<Vec<f64>> {
        match self.coords.as_ref()? {
            Coords::Cpu(v) => Some(v.clone()),
            Coords::Device(t) => Some(self.device_coords_to_f64(t)),
        }
    }

    /// Borrows the coordinate buffer without copying. `None` when the cloud
    /// has no coordinates *or* is device-resident (where no contiguous f64
    /// buffer exists); callers must fall back to [`Self::coords_f64`].
    pub fn coords_f64_slice(&self) -> Option<&[f64]> {
        match self.coords.as_ref()? {
            Coords::Cpu(v) => Some(v.as_slice()),
            Coords::Device(_) => None,
        }
    }

    fn device_coords_to_f64(&self, tensor: &Tensor<B, 2>) -> Vec<f64> {
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
        out
    }

    /// A single coordinate axis (`0=x, 1=y, 2=z`) as f8, or `None` if
    /// coordinates were never established.
    pub fn axis_f64(&self, axis: usize) -> Option<Vec<f64>> {
        match self.coords.as_ref()? {
            Coords::Cpu(v) => Some(v.as_chunks::<3>().0.iter().map(|c| c[axis]).collect()),
            Coords::Device(t) => {
                let rel: Vec<f32> = t
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
        }
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
        self.offset = offset;
        self.coords = Some(self.encode_coords(xyz, n));
        Ok(())
    }

    /// Builds the representation this cloud's device calls for: absolute f64
    /// on CPU, f32 relative to `offset` on an accelerator.
    fn encode_coords(&self, xyz: &[f64], n: usize) -> Coords {
        if backend::is_cpu_device(&self.device) {
            return Coords::Cpu(xyz.to_vec());
        }
        let offset = self.offset;
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
        Coords::Device(Box::new(Tensor::<B, 2>::from_data(
            TensorData::new(rel, [n, 3]),
            &self.device,
        )))
    }

    /// Sets one coordinate axis (`0=x, 1=y, 2=z`) in place. If this is the
    /// cloud's first coordinate write, the other two axes are established at
    /// zero.
    pub fn set_axis_f64(&mut self, axis: usize, values: &[f64]) -> Result<()> {
        let n = values.len();
        self.check_or_set_len(n)?;
        // Work in absolute f64 regardless of the current representation, so
        // the write is exact; re-encode for the device at the end.
        let mut abs = match self.coords.as_ref() {
            Some(_) => self
                .coords_f64()
                .expect("coords is Some, so coords_f64 yields a buffer"),
            None => vec![0.0f64; n * 3],
        };
        for (row, &v) in values.iter().enumerate() {
            abs[row * 3 + axis] = v;
        }
        self.coords = Some(self.encode_coords(&abs, n));
        Ok(())
    }

    // ---- dimension plane ------------------------------------------------

    pub fn has_dim(&self, name: &str) -> bool {
        if is_coordinate_name(name) {
            return self.coords.is_some();
        }
        self.dim_index(name).is_some()
    }

    pub fn dim_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.dims.len() + 3);
        if self.coords.is_some() {
            names.push("x".to_string());
            names.push("y".to_string());
            names.push("z".to_string());
        }
        names.extend(self.dims.iter().map(|(n, _)| n.clone()));
        names
    }

    pub fn dim_dtypes(&self) -> Vec<(String, DType)> {
        let mut out = Vec::with_capacity(self.dims.len() + 3);
        if self.coords.is_some() {
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

    /// Moves the cloud to `device`, re-encoding coordinates for the target's
    /// representation (absolute f64 on CPU, f32 relative to `offset` on an
    /// accelerator). Moving GPU -> CPU therefore widens back to f64, but the
    /// f32 rounding already incurred is not recovered.
    pub fn to_device(&self, device: DispatchDevice) -> Self {
        let mut cloned = self.clone();
        if self.coords.is_some() {
            let abs = self
                .coords_f64()
                .expect("coords is Some, so coords_f64 yields a buffer");
            cloned.device = device;
            cloned.coords = Some(cloned.encode_coords(&abs, self.len));
        } else {
            cloned.device = device;
        }
        cloned
    }

    /// Clones the cloud keeping only the rows in `selected`, gathering the
    /// coordinate plane and every dimension column by the same indices.
    /// `selected` must be in-bounds; callers pass it sorted ascending so the
    /// output preserves input order.
    pub(crate) fn with_gathered_rows(&self, selected: &[u32]) -> Result<Self> {
        let m = selected.len();
        let mut result = self.clone_without_coords();

        result.coords = match self.coords.as_ref() {
            None => None,
            Some(Coords::Cpu(v)) => {
                let mut out = Vec::with_capacity(m * 3);
                for &idx in selected {
                    let base = idx as usize * 3;
                    out.extend_from_slice(&v[base..base + 3]);
                }
                Some(Coords::Cpu(out))
            }
            Some(Coords::Device(t)) => {
                let rel: Vec<f32> = t
                    .to_data()
                    .to_vec::<f32>()
                    .expect("coordinate tensor is always f32");
                let mut out = Vec::with_capacity(m * 3);
                for &idx in selected {
                    let base = idx as usize * 3;
                    out.extend_from_slice(&rel[base..base + 3]);
                }
                Some(Coords::Device(Box::new(Tensor::<B, 2>::from_data(
                    TensorData::new(out, [m, 3]),
                    &self.device,
                ))))
            }
        };

        result.len = m;
        for (_, dim) in result.dims.iter_mut() {
            *dim = dim.gather(selected);
        }
        Ok(result)
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}
