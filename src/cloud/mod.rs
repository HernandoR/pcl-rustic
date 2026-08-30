//! `PointCloud`: the single mutation choke point for all dimension data
//! (ADR-0002, ADR-0003).
//!
//! Coordinates have three representations, picked by device placement and
//! the cloud's coordinate dtype (see [`CoordDtype`]):
//!
//! * **CPU-resident, f64 dtype** clouds hold absolute f64 coordinates in a
//!   plain `Vec`. This makes `x`/`y`/`z` exact (no f32 rounding), and it
//!   removes the f32 -> f64 conversion on every coordinate read.
//! * **CPU-resident, f32 dtype** clouds hold f32 values relative to
//!   `offset` -- the same representation the device uses -- so CPU and GPU
//!   behave identically under the default dtype, and readout skips the
//!   widening pass.
//! * **Device-resident** clouds hold f32 values relative to `offset`
//!   regardless of dtype, since wgpu has no f64. Precision then depends on
//!   the cloud's extent, which is why `offset` anchors near the data.
//!
//! The dtype also decides the numpy dtype coordinate reads produce: an f32
//! cloud hands out float32, skipping both the widening pass and half the
//! copy volume, which is most of the GPU materialization overhead.
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
use burn::tensor::backend::Backend as _;
use burn::tensor::{Tensor, TensorData};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, LazyLock};

/// The precision a cloud's coordinates are stored at and read out as.
/// Chosen per cloud at construction from [`default_coord_dtype`], never
/// changed afterwards; `to_device` preserves it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CoordDtype {
    F32,
    F64,
}

impl CoordDtype {
    pub fn numpy_name(self) -> &'static str {
        match self {
            CoordDtype::F32 => "float32",
            CoordDtype::F64 => "float64",
        }
    }

    /// Accepts the numpy names plus the Rust-style short forms.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "float32" | "f32" => Some(CoordDtype::F32),
            "float64" | "f64" => Some(CoordDtype::F64),
            _ => None,
        }
    }
}

/// Process-wide default coordinate dtype, torch-style: it decides the dtype
/// of *newly constructed* clouds only; existing clouds keep theirs.
///
/// f32 is the default because it makes CPU and GPU behave identically and
/// halves coordinate readout cost -- the f32 -> f64 widening pass was
/// measured to dominate GPU materialization (ADR-0002 measured the same on
/// CPU before its amendment). Workflows that need laspy-exact f64 (surveying
/// at UTM scale, LAS fidelity comparisons) opt back in globally via
/// `set_default_dtype("float64")` or `PCL_RUSTIC_DEFAULT_DTYPE=float64`.
///
/// The env var is read once, on first use. An invalid value cannot raise
/// (this can run inside `import pcl_rustic`, which must not fail -- ADR-0001
/// extends the same rule to probing), so it warns on stderr and falls back
/// to f32.
static DEFAULT_DTYPE: LazyLock<AtomicU8> = LazyLock::new(|| {
    let initial = match std::env::var("PCL_RUSTIC_DEFAULT_DTYPE") {
        Ok(raw) => CoordDtype::parse(&raw).unwrap_or_else(|| {
            eprintln!(
                "pcl_rustic: ignoring invalid PCL_RUSTIC_DEFAULT_DTYPE={raw:?} \
                 (expected 'float32' or 'float64'); defaulting to float32"
            );
            CoordDtype::F32
        }),
        Err(_) => CoordDtype::F32,
    };
    AtomicU8::new(initial as u8)
});

pub fn default_coord_dtype() -> CoordDtype {
    match DEFAULT_DTYPE.load(Ordering::Relaxed) {
        x if x == CoordDtype::F64 as u8 => CoordDtype::F64,
        _ => CoordDtype::F32,
    }
}

pub fn set_default_coord_dtype(dtype: CoordDtype) {
    DEFAULT_DTYPE.store(dtype as u8, Ordering::Relaxed);
}

/// How a cloud's coordinates are stored. See the module comment.
#[derive(Clone)]
pub(crate) enum Coords {
    /// Absolute f64, row-major `[x0,y0,z0,...]`, CPU-resident (f64 dtype).
    ///
    /// Behind an `Arc` for copy-on-write: Python-side zero-copy views hold a
    /// clone of this `Arc`, so the buffer they point at outlives any later
    /// mutation of the cloud -- mutating paths go through `Arc::make_mut`,
    /// which clones the buffer if (and only if) a view or another cloud
    /// still shares it. Views are therefore stable snapshots, never dangling
    /// and never spooky-mutating. `PointCloud::clone()` also becomes O(1)
    /// on this buffer as a side effect.
    CpuF64(Arc<Vec<f64>>),
    /// f32 relative to the cloud's `offset`, row-major, CPU-resident (f32
    /// dtype). Deliberately the device representation on host memory:
    /// precision does not change when a cloud moves to and from the GPU.
    CpuF32(Vec<f32>),
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
    /// Coordinate precision, fixed at construction from the global default.
    dtype: CoordDtype,
}

impl PointCloud {
    pub fn new() -> Self {
        let dtype = default_coord_dtype();
        // f64 clouds ingest CPU-resident even when the default device is a
        // GPU: the device representation is f32, so honoring the default
        // placement would round the coordinates at construction and make
        // the f64 opt-in meaningless on any GPU host. Rounding under f64
        // dtype only ever happens through an *explicit* `to_device`.
        let device = match dtype {
            CoordDtype::F64 => backend::cpu_device(),
            CoordDtype::F32 => backend::default_device(),
        };
        Self {
            coords: None,
            offset: [0.0; 3],
            scales: [0.001; 3],
            dims: Vec::new(),
            len: 0,
            device,
            dtype,
        }
    }

    pub fn dtype(&self) -> CoordDtype {
        self.dtype
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
            dtype: self.dtype,
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
    /// coordinates were never established. CPU-f64 clouds copy their buffer;
    /// relative representations widen f32 -> f64 and fold in `offset`
    /// (device-resident ones transfer first).
    pub fn coords_f64(&self) -> Option<Vec<f64>> {
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(v.as_ref().clone()),
            Coords::CpuF32(rel) => Some(self.rel_to_f64(rel)),
            Coords::Device(t) => Some(self.rel_to_f64(&Self::device_rel(t))),
        }
    }

    /// The shared absolute-f64 buffer, for zero-copy views: `None` unless
    /// the cloud is CPU-resident at f64 dtype. The other representations
    /// are relative, so a readout necessarily computes `offset + rel` --
    /// there is no absolute buffer to point a view at.
    pub fn coords_f64_arc(&self) -> Option<Arc<Vec<f64>>> {
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(Arc::clone(v)),
            Coords::CpuF32(_) | Coords::Device(_) => None,
        }
    }

    /// Row-major f4 *absolute* coordinates, or `None` if never established.
    /// The natural readout for f32 clouds: relative representations add
    /// `offset` without a widening pass (the sum is computed in f64 and
    /// rounded once, which is identical to what a wider readout narrowed
    /// afterwards would give).
    pub fn coords_f32(&self) -> Option<Vec<f32>> {
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(v.par_iter().map(|&x| x as f32).collect()),
            Coords::CpuF32(rel) => Some(self.rel_to_f32(rel)),
            Coords::Device(t) => {
                // The readback buffer is ours to overwrite: fold the offset
                // in place instead of allocating a second 12-bytes-per-point
                // buffer just to write `offset + rel` into it.
                let mut rel = Self::device_rel(t);
                let offset = self.offset;
                rel.par_chunks_mut(3).for_each(|r| {
                    r[0] = (offset[0] + r[0] as f64) as f32;
                    r[1] = (offset[1] + r[1] as f64) as f32;
                    r[2] = (offset[2] + r[2] as f64) as f32;
                });
                Some(rel)
            }
        }
    }

    /// Borrows the coordinate buffer without copying. `None` when the cloud
    /// has no coordinates *or* holds a relative representation (where no
    /// contiguous absolute-f64 buffer exists); callers must fall back to
    /// [`Self::coords_f64`].
    pub fn coords_f64_slice(&self) -> Option<&[f64]> {
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(v.as_slice()),
            Coords::CpuF32(_) | Coords::Device(_) => None,
        }
    }

    /// Absolute f64 coordinates as a `Cow`: borrowed straight from the
    /// cloud's buffer when it is CPU-f64 (zero-copy), materialized
    /// otherwise. The single accessor every read-only consumer (file
    /// writers, voxel grouping) should use, so the copy only ever happens
    /// when the representation forces it.
    pub fn coords_f64_cow(&self) -> Option<std::borrow::Cow<'_, [f64]>> {
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(std::borrow::Cow::Borrowed(v.as_slice())),
            Coords::CpuF32(_) | Coords::Device(_) => self.coords_f64().map(std::borrow::Cow::Owned),
        }
    }

    /// Reads a device tensor's relative-f32 buffer back to host.
    fn device_rel(tensor: &Tensor<B, 2>) -> Vec<f32> {
        tensor
            .to_data()
            .to_vec::<f32>()
            .expect("coordinate tensor is always f32")
    }

    /// offset + rel, widened to f64. Rayon-parallel: this pass dominated
    /// GPU materialization (measured ~250 ms of a 260 ms `.xyz` readback at
    /// 10M points as a serial push loop).
    fn rel_to_f64(&self, rel: &[f32]) -> Vec<f64> {
        let offset = self.offset;
        let mut out = vec![0.0f64; rel.len()];
        out.par_chunks_mut(3)
            .zip(rel.par_chunks(3))
            .for_each(|(o, r)| {
                o[0] = offset[0] + r[0] as f64;
                o[1] = offset[1] + r[1] as f64;
                o[2] = offset[2] + r[2] as f64;
            });
        out
    }

    /// offset + rel, rounded to f32 once at the end.
    fn rel_to_f32(&self, rel: &[f32]) -> Vec<f32> {
        let offset = self.offset;
        let mut out = vec![0.0f32; rel.len()];
        out.par_chunks_mut(3)
            .zip(rel.par_chunks(3))
            .for_each(|(o, r)| {
                o[0] = (offset[0] + r[0] as f64) as f32;
                o[1] = (offset[1] + r[1] as f64) as f32;
                o[2] = (offset[2] + r[2] as f64) as f32;
            });
        out
    }

    /// A single coordinate axis (`0=x, 1=y, 2=z`) as f8, or `None` if
    /// coordinates were never established.
    pub fn axis_f64(&self, axis: usize) -> Option<Vec<f64>> {
        let off = self.offset[axis];
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(v.as_chunks::<3>().0.par_iter().map(|c| c[axis]).collect()),
            Coords::CpuF32(rel) => Some(
                rel.as_chunks::<3>()
                    .0
                    .par_iter()
                    .map(|c| off + c[axis] as f64)
                    .collect(),
            ),
            Coords::Device(t) => Some(
                Self::device_rel(t)
                    .as_chunks::<3>()
                    .0
                    .par_iter()
                    .map(|c| off + c[axis] as f64)
                    .collect(),
            ),
        }
    }

    /// A single coordinate axis as f4 absolute values.
    pub fn axis_f32(&self, axis: usize) -> Option<Vec<f32>> {
        let off = self.offset[axis];
        match self.coords.as_ref()? {
            Coords::CpuF64(v) => Some(
                v.as_chunks::<3>()
                    .0
                    .par_iter()
                    .map(|c| c[axis] as f32)
                    .collect(),
            ),
            Coords::CpuF32(rel) => Some(
                rel.as_chunks::<3>()
                    .0
                    .par_iter()
                    .map(|c| (off + c[axis] as f64) as f32)
                    .collect(),
            ),
            Coords::Device(t) => Some(
                Self::device_rel(t)
                    .as_chunks::<3>()
                    .0
                    .par_iter()
                    .map(|c| (off + c[axis] as f64) as f32)
                    .collect(),
            ),
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

    /// Builds the representation this cloud's device and dtype call for:
    /// absolute f64 on CPU at f64 dtype, f32 relative to `offset` otherwise
    /// (host-side for CPU clouds, a tensor for accelerator ones).
    fn encode_coords(&self, xyz: &[f64], n: usize) -> Coords {
        if backend::is_cpu_device(&self.device) && self.dtype == CoordDtype::F64 {
            return Coords::CpuF64(Arc::new(xyz.to_vec()));
        }
        let offset = self.offset;
        let mut rel = vec![0.0f32; xyz.len()];
        rel.par_chunks_mut(3)
            .zip(xyz.par_chunks(3))
            .for_each(|(r, c)| {
                r[0] = (c[0] - offset[0]) as f32;
                r[1] = (c[1] - offset[1]) as f32;
                r[2] = (c[2] - offset[2]) as f32;
            });
        if backend::is_cpu_device(&self.device) {
            return Coords::CpuF32(rel);
        }
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
        let coord_dtype = match self.dtype {
            CoordDtype::F32 => DType::F32,
            CoordDtype::F64 => DType::F64,
        };
        let mut out = Vec::with_capacity(self.dims.len() + 3);
        if self.coords.is_some() {
            out.push(("x".to_string(), coord_dtype));
            out.push(("y".to_string(), coord_dtype));
            out.push(("z".to_string(), coord_dtype));
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

    /// Blocks until every queued device op on this cloud's coordinates has
    /// executed, *without* reading anything back. GPU tensor ops run
    /// asynchronously, so wall-clocking an op without this measures only
    /// submission; with it, compute is included but the PCIe readback and
    /// numpy materialization are not. No-op for CPU-resident clouds.
    pub fn synchronize(&self) {
        if let Some(Coords::Device(_)) = self.coords.as_ref() {
            // A sync failure means queued work failed; readback would
            // surface the same error later, so a benchmark sync point
            // treats it as fatal rather than silently timing nothing.
            B::sync(&self.device).expect("device sync failed");
        }
    }

    // ---- device -----------------------------------------------------------

    /// Moves the cloud to `device`, re-encoding coordinates for the target's
    /// representation. The f32-relative representations (CPU-f32 and device)
    /// share a layout, so moves between them copy the relative buffer as-is
    /// -- no precision change and no widen/narrow round trip. Only moves
    /// into or out of the CPU-f64 representation re-encode; GPU -> CPU-f64
    /// widens back to f64 but does not recover rounding already incurred.
    pub fn to_device(&self, device: DispatchDevice) -> Self {
        let mut cloned = self.clone();
        cloned.device = device.clone();
        let Some(coords) = self.coords.as_ref() else {
            return cloned;
        };
        let target_is_cpu = backend::is_cpu_device(&device);
        cloned.coords = Some(match (coords, target_is_cpu, self.dtype) {
            // CPU -> CPU keeps the representation bit-for-bit.
            (Coords::CpuF64(_) | Coords::CpuF32(_), true, _) => coords.clone(),
            // Relative f32 buffers move without re-encoding.
            (Coords::CpuF32(rel), false, _) => Coords::Device(Box::new(Tensor::<B, 2>::from_data(
                TensorData::new(rel.clone(), [self.len, 3]),
                &device,
            ))),
            (Coords::Device(t), true, CoordDtype::F32) => Coords::CpuF32(Self::device_rel(t)),
            (Coords::Device(t), false, _) => Coords::Device(Box::new(t.clone().to_device(&device))),
            // Everything else goes through absolute f64.
            _ => {
                let abs = self
                    .coords_f64()
                    .expect("coords is Some, so coords_f64 yields a buffer");
                cloned.encode_coords(&abs, self.len)
            }
        });
        cloned
    }

    /// Clones the cloud keeping only the rows in `selected`, gathering the
    /// coordinate plane and every dimension column by the same indices.
    /// `selected` must be in-bounds; callers pass it sorted ascending so the
    /// output preserves input order.
    pub(crate) fn with_gathered_rows(&self, selected: &[u32]) -> Result<Self> {
        let m = selected.len();
        let mut result = self.clone_without_coords();

        fn gather3<T: Copy>(v: &[T], selected: &[u32]) -> Vec<T> {
            let mut out = Vec::with_capacity(selected.len() * 3);
            for &idx in selected {
                let base = idx as usize * 3;
                out.extend_from_slice(&v[base..base + 3]);
            }
            out
        }

        result.coords = match self.coords.as_ref() {
            None => None,
            Some(Coords::CpuF64(v)) => Some(Coords::CpuF64(Arc::new(gather3(v, selected)))),
            Some(Coords::CpuF32(rel)) => Some(Coords::CpuF32(gather3(rel, selected))),
            Some(Coords::Device(t)) => {
                let rel = Self::device_rel(t);
                Some(Coords::Device(Box::new(Tensor::<B, 2>::from_data(
                    TensorData::new(gather3(&rel, selected), [m, 3]),
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
