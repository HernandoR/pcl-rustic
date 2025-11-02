#![allow(clippy::useless_conversion)]

use nalgebra::Matrix4;
use pcl_rustic_core::{hello_from_core, Point, TablePointCloud};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyfunction]
fn hello_from_bind() -> String {
    hello_from_core()
}

/// Python wrapper for Point struct
#[pyclass]
#[derive(Clone)]
struct PyPoint {
    inner: Point<f64>,
}

#[pymethods]
impl PyPoint {
    /// Create a new point with x, y, z coordinates
    #[new]
    #[pyo3(signature = (x, y, z, attributes=None))]
    fn new(x: f64, y: f64, z: f64, attributes: Option<HashMap<String, f64>>) -> Self {
        let inner = if let Some(attrs) = attributes {
            Point::with_attributes(x, y, z, attrs)
        } else {
            Point::new(x, y, z)
        };
        PyPoint { inner }
    }

    /// Get x coordinate
    #[getter]
    fn x(&self) -> f64 {
        self.inner.x
    }

    /// Get y coordinate
    #[getter]
    fn y(&self) -> f64 {
        self.inner.y
    }

    /// Get z coordinate
    #[getter]
    fn z(&self) -> f64 {
        self.inner.z
    }

    /// Set an attribute
    fn set_attribute(&mut self, key: String, value: f64) {
        self.inner.set_attribute(key, value);
    }

    /// Get an attribute value
    fn get_attribute(&self, key: &str) -> Option<f64> {
        self.inner.get_attribute(key)
    }

    /// Get all attributes as a dictionary
    #[getter]
    fn attributes(&self) -> HashMap<String, f64> {
        self.inner.attributes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Point(x={}, y={}, z={}, attributes={:?})",
            self.inner.x, self.inner.y, self.inner.z, self.inner.attributes
        )
    }
}

/// Python wrapper for TablePointCloud struct
#[pyclass]
#[derive(Clone)]
struct PyTablePointCloud {
    inner: TablePointCloud,
}

#[pymethods]
impl PyTablePointCloud {
    /// Create a new empty TablePointCloud
    #[new]
    fn new() -> PyResult<Self> {
        let inner = TablePointCloud::new().map_err(|e| {
            PyValueError::new_err(format!("Failed to create TablePointCloud: {}", e))
        })?;
        Ok(PyTablePointCloud { inner })
    }

    /// Create a TablePointCloud from x, y, z coordinate lists
    #[staticmethod]
    fn from_xyz(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> PyResult<Self> {
        let inner = TablePointCloud::from_xyz(x, y, z)
            .map_err(|e| PyValueError::new_err(format!("Failed to create from xyz: {}", e)))?;
        Ok(PyTablePointCloud { inner })
    }

    /// Create a TablePointCloud from a list of PyPoint objects
    #[staticmethod]
    fn from_points(points: Vec<PyPoint>) -> PyResult<Self> {
        let rust_points: Vec<Point<f64>> = points.into_iter().map(|p| p.inner).collect();
        let inner = TablePointCloud::from_points(rust_points)
            .map_err(|e| PyValueError::new_err(format!("Failed to create from points: {}", e)))?;
        Ok(PyTablePointCloud { inner })
    }

    /// Get the number of points in the cloud
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the point cloud is empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get x coordinates as a list
    fn x(&self) -> PyResult<Vec<f64>> {
        self.inner
            .x()
            .map_err(|e| PyValueError::new_err(format!("Failed to get x: {}", e)))
    }

    /// Get y coordinates as a list
    fn y(&self) -> PyResult<Vec<f64>> {
        self.inner
            .y()
            .map_err(|e| PyValueError::new_err(format!("Failed to get y: {}", e)))
    }

    /// Get z coordinates as a list
    fn z(&self) -> PyResult<Vec<f64>> {
        self.inner
            .z()
            .map_err(|e| PyValueError::new_err(format!("Failed to get z: {}", e)))
    }

    /// Add an attribute column to the point cloud
    fn add_attribute(&mut self, name: &str, values: Vec<f64>) -> PyResult<()> {
        self.inner
            .add_attribute(name, values)
            .map_err(|e| PyValueError::new_err(format!("Failed to add attribute: {}", e)))
    }

    /// Get a point at a specific index
    fn get_point(&self, index: usize) -> PyResult<PyPoint> {
        let point = self
            .inner
            .get_point(index)
            .map_err(|e| PyValueError::new_err(format!("Failed to get point: {}", e)))?;
        Ok(PyPoint { inner: point })
    }

    /// Convert to a list of PyPoint objects
    fn to_points(&self) -> PyResult<Vec<PyPoint>> {
        let points = self
            .inner
            .to_points()
            .map_err(|e| PyValueError::new_err(format!("Failed to convert to points: {}", e)))?;
        Ok(points.into_iter().map(|p| PyPoint { inner: p }).collect())
    }

    /// Transform the point cloud using a 4x4 transformation matrix
    /// The matrix should be provided as a flat list of 16 values in row-major order
    fn transform(&self, matrix: Vec<f64>) -> PyResult<Self> {
        if matrix.len() != 16 {
            return Err(PyValueError::new_err(
                "Transformation matrix must have exactly 16 elements",
            ));
        }

        // Create Matrix4 from flat array (row-major)
        let transform_matrix = Matrix4::from_row_slice(&matrix);

        let transformed = self
            .inner
            .transform(&transform_matrix)
            .map_err(|e| PyValueError::new_err(format!("Failed to transform: {}", e)))?;

        Ok(PyTablePointCloud { inner: transformed })
    }

    fn __repr__(&self) -> String {
        format!("TablePointCloud(len={})", self.inner.len())
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bind, m)?)?;
    m.add_class::<PyPoint>()?;
    m.add_class::<PyTablePointCloud>()?;
    Ok(())
}
