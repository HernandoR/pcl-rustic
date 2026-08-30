//! The numpy <-> Rust boundary (ADR-0002 "Getter / setter semantics"):
//! setters require the pinned dtype exactly (same-kind coercion happens in
//! the `pcl_rustic` Python wrapper, not here); getters always copy into a
//! fresh numpy array.

use crate::cloud::dims::{DType, DimArray};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Base object for zero-copy coordinate views: holds an `Arc` clone of the
/// cloud's f64 buffer, so the numpy array's memory stays alive however long
/// the view outlives the cloud, and the cloud's own mutations copy-on-write
/// away from it rather than into it.
#[pyclass(name = "_CoordsOwner", frozen)]
pub struct CoordsOwner(#[allow(dead_code)] Arc<Vec<f64>>);

/// Builds a read-only, zero-copy `(n, 3)` f8 numpy view over the cloud's
/// shared coordinate buffer. Read-only because the buffer may be shared
/// with the cloud itself and with other views: writes through a view would
/// alias unpredictably (before/after a copy-on-write split), so mutation
/// stays with the cloud's own API.
pub fn xyz_view_pyobject(py: Python<'_>, buf: Arc<Vec<f64>>, n: usize) -> PyResult<Py<PyAny>> {
    debug_assert_eq!(buf.len(), n * 3);
    let view = numpy::ndarray::ArrayView2::from_shape((n, 3), &buf[..])
        .expect("coordinate buffer always has exactly 3n elements");
    let owner = Bound::new(py, CoordsOwner(Arc::clone(&buf)))?;
    // SAFETY: `owner` holds an `Arc` to the same allocation `view` borrows,
    // and is registered as the numpy array's base object, so the data
    // outlives the array. The buffer behind a shared `Arc<Vec>` is never
    // reallocated: every mutating path goes through `Arc::make_mut`, which
    // replaces the allocation rather than growing it.
    let array = unsafe { PyArray2::borrow_from_array(&view, owner.into_any()) };
    // Clear WRITEABLE (numpy arrays default to writable even over borrowed
    // data); readers who need a mutable array take a .copy() explicitly.
    unsafe {
        (*array.as_array_ptr()).flags &= !numpy::npyffi::NPY_ARRAY_WRITEABLE;
    }
    Ok(array.into_any().unbind())
}

/// Downcasts `obj` to a 1D numpy array of exactly dtype `T`, or raises a
/// `TypeError` naming the dtype the caller must supply.
fn as_array1<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    expected: &str,
) -> PyResult<PyReadonlyArray1<'py, T>> {
    obj.cast::<PyArray1<T>>()
        .map(|a| a.readonly())
        .map_err(|_| {
            PyTypeError::new_err(format!("expected a 1D numpy array of dtype '{expected}'"))
        })
}

/// Downcasts `obj` to a 2D numpy array of exactly dtype `T`.
fn as_array2<'py, T: numpy::Element>(
    obj: &Bound<'py, PyAny>,
    expected: &str,
) -> PyResult<PyReadonlyArray2<'py, T>> {
    obj.cast::<PyArray2<T>>()
        .map(|a| a.readonly())
        .map_err(|_| {
            PyTypeError::new_err(format!("expected a 2D numpy array of dtype '{expected}'"))
        })
}

/// Reads a 1D array matching `dtype` exactly into a `DimArray` (`set_dim`).
pub fn dim_array_from_pyobject(obj: &Bound<PyAny>, dtype: DType) -> PyResult<DimArray> {
    Ok(match dtype {
        DType::U8 => DimArray::U8(
            as_array1::<u8>(obj, "uint8")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::U16 => DimArray::U16(
            as_array1::<u16>(obj, "uint16")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::U32 => DimArray::U32(
            as_array1::<u32>(obj, "uint32")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::U64 => DimArray::U64(
            as_array1::<u64>(obj, "uint64")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::I8 => DimArray::I8(
            as_array1::<i8>(obj, "int8")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::I16 => DimArray::I16(
            as_array1::<i16>(obj, "int16")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::I32 => DimArray::I32(
            as_array1::<i32>(obj, "int32")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::I64 => DimArray::I64(
            as_array1::<i64>(obj, "int64")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::F32 => DimArray::F32(
            as_array1::<f32>(obj, "float32")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::F64 => DimArray::F64(
            as_array1::<f64>(obj, "float64")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
        DType::Bool => DimArray::Bool(
            as_array1::<bool>(obj, "bool")?
                .as_array()
                .iter()
                .copied()
                .collect(),
        ),
    })
}

/// Builds a fresh (copied) numpy array from a `DimArray` (`get_dim`).
pub fn dim_array_to_pyobject(py: Python<'_>, dim: &DimArray) -> Py<PyAny> {
    match dim {
        DimArray::U8(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::U16(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::U32(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::U64(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::I8(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::I16(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::I32(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::I64(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::F32(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::F64(v) => v.clone().into_pyarray(py).into_any().unbind(),
        DimArray::Bool(v) => v.clone().into_pyarray(py).into_any().unbind(),
    }
}

fn check_ncols3(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyTypeError::new_err(format!(
            "expected an (N, 3) numpy array, got shape {shape:?}"
        )));
    }
    Ok(())
}

/// Reads an `(N, 3)` array of `f4|f8|i4|i8` for `from_xyz`, widening to f8.
pub fn from_xyz_pyobject(obj: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = obj.cast::<PyArray2<f64>>() {
        let a = a.readonly();
        check_ncols3(a.shape())?;
        return Ok(a.as_array().iter().copied().collect());
    }
    if let Ok(a) = obj.cast::<PyArray2<f32>>() {
        let a = a.readonly();
        check_ncols3(a.shape())?;
        return Ok(a.as_array().iter().map(|&v| v as f64).collect());
    }
    if let Ok(a) = obj.cast::<PyArray2<i32>>() {
        let a = a.readonly();
        check_ncols3(a.shape())?;
        return Ok(a.as_array().iter().map(|&v| v as f64).collect());
    }
    if let Ok(a) = obj.cast::<PyArray2<i64>>() {
        let a = a.readonly();
        check_ncols3(a.shape())?;
        return Ok(a.as_array().iter().map(|&v| v as f64).collect());
    }
    Err(PyTypeError::new_err(
        "from_xyz expects an (N, 3) numpy array of dtype float32, float64, int32, or int64",
    ))
}

/// Reads a strict `(N, 3)` f8 numpy array (`set_dim("xyz", ...)`).
pub fn xyz_from_pyobject(obj: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    let array = as_array2::<f64>(obj, "float64")?;
    check_ncols3(array.shape())?;
    Ok(array.as_array().iter().copied().collect())
}

/// Builds a fresh `(n, 3)` f8 numpy array from row-major data.
pub fn xyz_to_pyobject(py: Python<'_>, flat: Vec<f64>, n: usize) -> Py<PyAny> {
    let array = numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
        .expect("flat xyz buffer always has exactly 3n elements");
    PyArray2::from_owned_array(py, array).into_any().unbind()
}

/// Builds a fresh `(n, 3)` f4 numpy array from row-major data (the readout
/// for float32-dtype clouds).
pub fn xyz32_to_pyobject(py: Python<'_>, flat: Vec<f32>, n: usize) -> Py<PyAny> {
    let array = numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
        .expect("flat xyz buffer always has exactly 3n elements");
    PyArray2::from_owned_array(py, array).into_any().unbind()
}

/// Reads a 1D f8 numpy array (`set_dim("x"/"y"/"z", ...)`).
pub fn axis_from_pyobject(obj: &Bound<PyAny>) -> PyResult<Vec<f64>> {
    Ok(as_array1::<f64>(obj, "float64")?
        .as_array()
        .iter()
        .copied()
        .collect())
}

pub fn axis_to_pyobject(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

pub fn axis32_to_pyobject(py: Python<'_>, values: Vec<f32>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

/// Reads a 3x3 or 4x4 f8 matrix, returning its flat row-major data and side
/// length (`transform`).
pub fn matrix_from_pyobject(obj: &Bound<PyAny>) -> PyResult<(Vec<f64>, usize)> {
    let array = as_array2::<f64>(obj, "float64")?;
    let shape = array.shape();
    if shape.len() != 2 || shape[0] != shape[1] || (shape[0] != 3 && shape[0] != 4) {
        return Err(PyTypeError::new_err(format!(
            "expected a (3, 3) or (4, 4) numpy array, got shape {shape:?}"
        )));
    }
    Ok((array.as_array().iter().copied().collect(), shape[0]))
}

/// Reads a fixed 3x3 f8 rotation matrix (`rigid_transform`).
pub fn rotation_from_pyobject(obj: &Bound<PyAny>) -> PyResult<[[f64; 3]; 3]> {
    let array = as_array2::<f64>(obj, "float64")?;
    if array.shape() != [3, 3] {
        return Err(PyTypeError::new_err(format!(
            "expected a (3, 3) numpy array, got shape {:?}",
            array.shape()
        )));
    }
    let a = array.as_array();
    Ok([
        [a[[0, 0]], a[[0, 1]], a[[0, 2]]],
        [a[[1, 0]], a[[1, 1]], a[[1, 2]]],
        [a[[2, 0]], a[[2, 1]], a[[2, 2]]],
    ])
}

/// Reads a fixed length-3 f8 translation vector (`rigid_transform`).
pub fn translation_from_pyobject(obj: &Bound<PyAny>) -> PyResult<[f64; 3]> {
    let values = axis_from_pyobject(obj)?;
    if values.len() != 3 {
        return Err(PyTypeError::new_err(format!(
            "expected a length-3 numpy array, got length {}",
            values.len()
        )));
    }
    Ok([values[0], values[1], values[2]])
}
