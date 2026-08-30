//! PyO3 bindings root: the small top-level surface (`DownsampleStrategy`,
//! `available_devices`, `default_device`, `device_report`) plus module wiring
//! for `cloud` and `convert`. `PointCloud` itself lives in `cloud.rs`.

pub mod cloud;
pub mod convert;

pub use cloud::PyPointCloud;

use crate::backend;
use crate::cloud::{default_coord_dtype, set_default_coord_dtype, CoordDtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Voxel downsampling strategy codes, exposed as class attributes rather
/// than a Python enum to keep the PyO3 surface minimal (ADR-0003). The
/// values are the `strategy` argument `PointCloud.voxel_downsample` expects.
#[pyclass(name = "DownsampleStrategy")]
pub struct PyDownsampleStrategy;

#[pymethods]
impl PyDownsampleStrategy {
    #[classattr]
    const NEAREST_TO_CENTROID: i64 = 0;
    #[classattr]
    const RANDOM: i64 = 1;
}

#[pyfunction]
pub fn available_devices() -> Vec<String> {
    backend::available_devices()
}

#[pyfunction]
pub fn default_device() -> String {
    backend::device_name(&backend::default_device()).to_string()
}

/// Returned as a list of `(name, status)` pairs rather than a dict so the
/// roster order survives the boundary; the Python wrapper builds the dict.
#[pyfunction]
pub fn device_report() -> Vec<(String, String)> {
    backend::device_report()
}

/// Sets the process-wide default coordinate dtype for *newly constructed*
/// clouds (torch-style semantics; existing clouds keep theirs). The Python
/// wrapper normalizes numpy dtype objects to a name before calling this.
#[pyfunction]
pub fn set_default_dtype(dtype: &str) -> PyResult<()> {
    let parsed = CoordDtype::parse(dtype).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unsupported coordinate dtype '{dtype}'; expected 'float32' or 'float64'"
        ))
    })?;
    set_default_coord_dtype(parsed);
    Ok(())
}

#[pyfunction]
pub fn get_default_dtype() -> String {
    default_coord_dtype().numpy_name().to_string()
}
