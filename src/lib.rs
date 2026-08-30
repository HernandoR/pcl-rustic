//! `_core`: the PyO3 extension module for pcl-rustic (ADR-0003). This file
//! only registers types and functions; all behavior lives in `cloud`, `io`,
//! `backend`, and `py`.

mod backend;
mod cloud;
mod error;
mod io;
mod py;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py::PyPointCloud>()?;
    m.add_class::<py::PyDownsampleStrategy>()?;
    m.add_function(wrap_pyfunction!(py::available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(py::default_device, m)?)?;
    m.add_function(wrap_pyfunction!(py::device_report, m)?)?;
    m.add_function(wrap_pyfunction!(py::set_default_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(py::get_default_dtype, m)?)?;
    Ok(())
}
