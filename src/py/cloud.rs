//! `PointCloud` PyO3 bindings, implementing the ADR-0003 contract exactly.

use crate::backend;
use crate::cloud::voxel::DownsampleStrategy;
use crate::cloud::PointCloud;
use crate::io::ColumnMap;
use crate::py::convert;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass(name = "PointCloud")]
pub struct PyPointCloud {
    pub(crate) inner: PointCloud,
}

fn columns_from_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Option<ColumnMap>> {
    let Some(dict) = dict else { return Ok(None) };
    let mut map = HashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        let value: String = v.extract()?;
        map.insert(key, value);
    }
    Ok(Some(map))
}

fn delimiter_byte(delimiter: &str) -> PyResult<u8> {
    if delimiter.len() != 1 || !delimiter.is_ascii() {
        return Err(PyTypeError::new_err(format!(
            "delimiter must be a single ASCII character, got '{delimiter}'"
        )));
    }
    Ok(delimiter.as_bytes()[0])
}

fn no_coordinates() -> PyErr {
    PyKeyError::new_err("point cloud has no coordinates")
}

#[pymethods]
impl PyPointCloud {
    #[new]
    fn new() -> Self {
        Self {
            inner: PointCloud::new(),
        }
    }

    #[staticmethod]
    fn from_xyz(xyz: &Bound<'_, PyAny>) -> PyResult<Self> {
        let flat = convert::from_xyz_pyobject(xyz)?;
        let mut inner = PointCloud::new();
        inner.set_coords_f64(&flat).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (path, columns=None, delimiter=","))]
    fn read(path: &str, columns: Option<&Bound<'_, PyDict>>, delimiter: &str) -> PyResult<Self> {
        let columns = columns_from_dict(columns)?;
        let delimiter = delimiter_byte(delimiter)?;
        let inner =
            PointCloud::read_file(path, columns.as_ref(), delimiter).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (path, columns=None, delimiter=","))]
    fn write(
        &self,
        path: &str,
        columns: Option<&Bound<'_, PyDict>>,
        delimiter: &str,
    ) -> PyResult<()> {
        let columns = columns_from_dict(columns)?;
        let delimiter = delimiter_byte(delimiter)?;
        self.inner
            .write_file(path, columns.as_ref(), delimiter)
            .map_err(PyErr::from)
    }

    fn get_dim(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        match name {
            "x" => Ok(convert::axis_to_pyobject(
                py,
                self.inner.axis_f64(0).ok_or_else(no_coordinates)?,
            )),
            "y" => Ok(convert::axis_to_pyobject(
                py,
                self.inner.axis_f64(1).ok_or_else(no_coordinates)?,
            )),
            "z" => Ok(convert::axis_to_pyobject(
                py,
                self.inner.axis_f64(2).ok_or_else(no_coordinates)?,
            )),
            "xyz" => {
                let flat = self.inner.coords_f64().ok_or_else(no_coordinates)?;
                Ok(convert::xyz_to_pyobject(py, flat, self.inner.len()))
            }
            _ => {
                let dim = self
                    .inner
                    .dim(name)
                    .ok_or_else(|| PyKeyError::new_err(format!("no such dimension '{name}'")))?;
                Ok(convert::dim_array_to_pyobject(py, dim))
            }
        }
    }

    fn set_dim(&mut self, name: &str, values: &Bound<'_, PyAny>) -> PyResult<()> {
        match name {
            "x" => self
                .inner
                .set_axis_f64(0, &convert::axis_from_pyobject(values)?)
                .map_err(PyErr::from),
            "y" => self
                .inner
                .set_axis_f64(1, &convert::axis_from_pyobject(values)?)
                .map_err(PyErr::from),
            "z" => self
                .inner
                .set_axis_f64(2, &convert::axis_from_pyobject(values)?)
                .map_err(PyErr::from),
            "xyz" => self
                .inner
                .set_coords_f64(&convert::xyz_from_pyobject(values)?)
                .map_err(PyErr::from),
            _ => {
                let dtype = self.inner.required_dim_dtype(name).ok_or_else(|| {
                    PyKeyError::new_err(format!(
                        "unknown dimension '{name}'; call add_extra_dim() first to declare its dtype"
                    ))
                })?;
                let dim = convert::dim_array_from_pyobject(values, dtype)?;
                self.inner.set_dim_array(name, dim).map_err(PyErr::from)
            }
        }
    }

    fn has_dim(&self, name: &str) -> bool {
        self.inner.has_dim(name)
    }

    fn remove_dim(&mut self, name: &str) -> PyResult<()> {
        self.inner.remove_dim(name).map_err(PyErr::from)
    }

    fn dim_names(&self) -> Vec<String> {
        self.inner.dim_names()
    }

    fn dim_dtypes(&self) -> HashMap<String, String> {
        self.inner
            .dim_dtypes()
            .into_iter()
            .map(|(name, dtype)| (name, dtype.numpy_name().to_string()))
            .collect()
    }

    fn add_extra_dim(&mut self, name: &str, dtype: &str) -> PyResult<()> {
        let dtype = crate::cloud::dims::DType::parse_extra(dtype).map_err(PyErr::from)?;
        self.inner.add_extra_dim(name, dtype).map_err(PyErr::from)
    }

    fn transform(&self, matrix: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (flat, side) = convert::matrix_from_pyobject(matrix)?;
        let inner = self.inner.transform(&flat, side).map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    fn rigid_transform(
        &self,
        rotation: &Bound<'_, PyAny>,
        translation: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let rotation = convert::rotation_from_pyobject(rotation)?;
        let translation = convert::translation_from_pyobject(translation)?;
        let inner = self
            .inner
            .rigid_transform(&rotation, &translation)
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (voxel_size, strategy=0, seed=None))]
    fn voxel_downsample(
        &self,
        voxel_size: f64,
        strategy: i64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let strategy = match strategy {
            0 => DownsampleStrategy::NearestToCentroid,
            1 => DownsampleStrategy::Random { seed },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown downsample strategy {other}"
                )))
            }
        };
        let inner = self
            .inner
            .voxel_downsample(voxel_size, strategy)
            .map_err(PyErr::from)?;
        Ok(Self { inner })
    }

    fn to_device(&self, device: &str) -> PyResult<Self> {
        let device = backend::parse_device_name(device).map_err(PyErr::from)?;
        Ok(Self {
            inner: self.inner.to_device(device),
        })
    }

    #[getter]
    fn device(&self) -> String {
        backend::device_name(self.inner.device()).to_string()
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PointCloud(len={}, dims={:?}, device='{}')",
            self.inner.len(),
            self.inner.dim_names(),
            backend::device_name(self.inner.device())
        )
    }
}
