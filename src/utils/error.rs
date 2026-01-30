/// 统一异常类型定义和Rust→Python异常转换
use pyo3::exceptions::*;
use pyo3::PyErr;
use thiserror::Error;

/// PCL Rustic全局异常类型
#[derive(Error, Debug)]
pub enum PointCloudError {
    #[error("IO错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("格式解析失败: {0}")]
    ParseError(String),

    #[error("维度不匹配: 期望 {expected}，实际 {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("张量形状错误: {0}")]
    TensorShapeError(String),

    #[error("矩阵操作错误: {0}")]
    MatrixError(String),

    #[error("文件不存在: {0}")]
    FileNotFound(String),

    #[error("格式不支持: {0}")]
    UnsupportedFormat(String),

    #[error("参数无效: {0}")]
    InvalidParameter(String),

    #[error("下采样失败: {0}")]
    DownsampleError(String),

    #[error("内存不足")]
    MemoryError,

    #[error("转换失败: {0}")]
    ConversionError(String),

    #[error("{0}")]
    Other(String),
}

impl From<PointCloudError> for PyErr {
    fn from(err: PointCloudError) -> PyErr {
        match err {
            PointCloudError::IoError(e) => PyIOError::new_err(e.to_string()),
            PointCloudError::FileNotFound(path) => {
                PyFileNotFoundError::new_err(format!("文件不存在: {}", path))
            }
            PointCloudError::DimensionMismatch { expected, actual } => {
                PyValueError::new_err(format!("维度不匹配: 期望 {}，实际 {}", expected, actual))
            }
            PointCloudError::ParseError(msg) => {
                PyValueError::new_err(format!("格式解析失败: {}", msg))
            }
            PointCloudError::TensorShapeError(msg) => {
                PyValueError::new_err(format!("张量形状错误: {}", msg))
            }
            PointCloudError::MatrixError(msg) => {
                PyValueError::new_err(format!("矩阵操作错误: {}", msg))
            }
            PointCloudError::UnsupportedFormat(fmt) => {
                PyValueError::new_err(format!("格式不支持: {}", fmt))
            }
            PointCloudError::InvalidParameter(msg) => {
                PyValueError::new_err(format!("参数无效: {}", msg))
            }
            PointCloudError::DownsampleError(msg) => {
                PyValueError::new_err(format!("下采样失败: {}", msg))
            }
            PointCloudError::MemoryError => PyMemoryError::new_err("内存不足"),
            PointCloudError::ConversionError(msg) => {
                PyValueError::new_err(format!("转换失败: {}", msg))
            }
            PointCloudError::Other(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<String> for PointCloudError {
    fn from(msg: String) -> Self {
        PointCloudError::Other(msg)
    }
}

impl From<&str> for PointCloudError {
    fn from(msg: &str) -> Self {
        PointCloudError::Other(msg.to_string())
    }
}

pub type Result<T> = std::result::Result<T, PointCloudError>;
