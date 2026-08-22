//! Unified crate error type and its mapping onto built-in Python exceptions
//! at the PyO3 boundary.

use pyo3::exceptions::{PyKeyError, PyOSError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

/// Crate-wide error type. Each variant maps to a specific Python exception
/// type (see `impl From<Error> for PyErr` below); nothing in this crate
/// raises a bare `RuntimeError`.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid argument value: bad shape, bad range, unknown format, unknown
    /// device name, ...
    #[error("{0}")]
    Value(String),
    /// Wrong Python/numpy type where a specific one was required.
    #[error("{0}")]
    Type(String),
    /// Unknown dimension name.
    #[error("{0}")]
    Key(String),
    /// Numeric value out of range for the target dtype.
    #[error("{0}")]
    Overflow(String),
    /// File system / codec failure (LAS, CSV, Parquet, or plain I/O).
    #[error("{0}")]
    Os(String),
}

impl Error {
    pub fn value(msg: impl Into<String>) -> Self {
        Self::Value(msg.into())
    }

    pub fn type_(msg: impl Into<String>) -> Self {
        Self::Type(msg.into())
    }

    pub fn key(msg: impl Into<String>) -> Self {
        Self::Key(msg.into())
    }

    pub fn overflow(msg: impl Into<String>) -> Self {
        Self::Overflow(msg.into())
    }

    pub fn os(msg: impl Into<String>) -> Self {
        Self::Os(msg.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Os(err.to_string())
    }
}

impl From<las::Error> for Error {
    fn from(err: las::Error) -> Self {
        Self::Os(format!("LAS error: {err}"))
    }
}

impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Self {
        Self::Os(format!("CSV error: {err}"))
    }
}

impl From<arrow::error::ArrowError> for Error {
    fn from(err: arrow::error::ArrowError) -> Self {
        Self::Os(format!("Arrow error: {err}"))
    }
}

impl From<parquet::errors::ParquetError> for Error {
    fn from(err: parquet::errors::ParquetError) -> Self {
        Self::Os(format!("Parquet error: {err}"))
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::Value(msg) => PyValueError::new_err(msg),
            Error::Type(msg) => PyTypeError::new_err(msg),
            Error::Key(msg) => PyKeyError::new_err(msg),
            Error::Overflow(msg) => PyOverflowError::new_err(msg),
            Error::Os(msg) => PyOSError::new_err(msg),
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
