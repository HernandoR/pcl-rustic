//! PyO3 bindings root: the small top-level surface (`DownsampleStrategy`,
//! `available_devices`, `default_device`) plus module wiring for `cloud` and
//! `convert`. `PointCloud` itself lives in `cloud.rs`.

pub mod cloud;
pub mod convert;

pub use cloud::PyPointCloud;

use crate::backend;
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
