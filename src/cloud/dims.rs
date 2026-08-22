//! Dtype tags, the `DimArray` column store, and the standard-dimension
//! registry (ADR-0002).

use crate::error::{Error, Result};

/// Scalar element dtype tag shared by `DimArray`, the standard-dimension
/// registry, and the extra-dimension declaration set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Bool,
}

impl DType {
    /// The numpy dtype name reported by `dim_dtypes()`.
    pub fn numpy_name(self) -> &'static str {
        match self {
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::Bool => "bool",
        }
    }

    /// Parses an extra-dimension dtype spec, accepting both laspy-style
    /// letter codes (`u1`..`f8`) and numpy dtype names. `bool` is
    /// intentionally not part of the allowed set (ADR-0002).
    pub fn parse_extra(spec: &str) -> Result<DType> {
        let dtype = match spec {
            "u1" | "uint8" => DType::U8,
            "u2" | "uint16" => DType::U16,
            "u4" | "uint32" => DType::U32,
            "u8" | "uint64" => DType::U64,
            "i1" | "int8" => DType::I8,
            "i2" | "int16" => DType::I16,
            "i4" | "int32" => DType::I32,
            "i8" | "int64" => DType::I64,
            "f4" | "float32" => DType::F32,
            "f8" | "float64" => DType::F64,
            other => {
                return Err(Error::value(format!(
                    "unsupported extra-dimension dtype '{other}'; expected one of \
                     u1, u2, u4, u8, i1, i2, i4, i8, f4, f8 (or the equivalent numpy names)"
                )))
            }
        };
        Ok(dtype)
    }
}

/// Column store for a single non-coordinate dimension (ADR-0002).
/// Coordinate data (`x`/`y`/`z`) never lives here -- it rides the tensor
/// plane in `cloud::PointCloud`.
#[derive(Clone, Debug)]
pub enum DimArray {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
}

impl DimArray {
    pub fn len(&self) -> usize {
        match self {
            DimArray::U8(v) => v.len(),
            DimArray::U16(v) => v.len(),
            DimArray::U32(v) => v.len(),
            DimArray::U64(v) => v.len(),
            DimArray::I8(v) => v.len(),
            DimArray::I16(v) => v.len(),
            DimArray::I32(v) => v.len(),
            DimArray::I64(v) => v.len(),
            DimArray::F32(v) => v.len(),
            DimArray::F64(v) => v.len(),
            DimArray::Bool(v) => v.len(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            DimArray::U8(_) => DType::U8,
            DimArray::U16(_) => DType::U16,
            DimArray::U32(_) => DType::U32,
            DimArray::U64(_) => DType::U64,
            DimArray::I8(_) => DType::I8,
            DimArray::I16(_) => DType::I16,
            DimArray::I32(_) => DType::I32,
            DimArray::I64(_) => DType::I64,
            DimArray::F32(_) => DType::F32,
            DimArray::F64(_) => DType::F64,
            DimArray::Bool(_) => DType::Bool,
        }
    }

    /// Builds a zero-filled column of the given dtype and length.
    pub fn zeros(len: usize, dtype: DType) -> DimArray {
        match dtype {
            DType::U8 => DimArray::U8(vec![0; len]),
            DType::U16 => DimArray::U16(vec![0; len]),
            DType::U32 => DimArray::U32(vec![0; len]),
            DType::U64 => DimArray::U64(vec![0; len]),
            DType::I8 => DimArray::I8(vec![0; len]),
            DType::I16 => DimArray::I16(vec![0; len]),
            DType::I32 => DimArray::I32(vec![0; len]),
            DType::I64 => DimArray::I64(vec![0; len]),
            DType::F32 => DimArray::F32(vec![0.0; len]),
            DType::F64 => DimArray::F64(vec![0.0; len]),
            DType::Bool => DimArray::Bool(vec![false; len]),
        }
    }

    /// Gathers rows at `indices`, producing a new column of the same dtype.
    /// Used by voxel downsampling to keep every dimension in sync with the
    /// selected coordinate rows.
    pub fn gather(&self, indices: &[u32]) -> DimArray {
        match self {
            DimArray::U8(v) => DimArray::U8(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::U16(v) => DimArray::U16(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::U32(v) => DimArray::U32(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::U64(v) => DimArray::U64(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::I8(v) => DimArray::I8(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::I16(v) => DimArray::I16(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::I32(v) => DimArray::I32(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::I64(v) => DimArray::I64(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::F32(v) => DimArray::F32(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::F64(v) => DimArray::F64(indices.iter().map(|&i| v[i as usize]).collect()),
            DimArray::Bool(v) => DimArray::Bool(indices.iter().map(|&i| v[i as usize]).collect()),
        }
    }
}

/// The ADR-0002 standard-dimension table, excluding `x`/`y`/`z` (handled by
/// the tensor coordinate plane in `cloud::PointCloud`, see `dim_dtypes`
/// there for how they are reported).
pub fn standard_dim_dtype(name: &str) -> Option<DType> {
    use DType::*;
    Some(match name {
        "intensity" => U16,
        "return_number" | "number_of_returns" | "scanner_channel" | "classification"
        | "user_data" => U8,
        "synthetic"
        | "key_point"
        | "withheld"
        | "overlap"
        | "scan_direction_flag"
        | "edge_of_flight_line" => Bool,
        "scan_angle" => I16,
        "point_source_id" => U16,
        "gps_time" => F64,
        "red" | "green" | "blue" | "nir" => U16,
        _ => return None,
    })
}

/// True for the three reserved coordinate names, handled outside `DimArray`.
pub fn is_coordinate_name(name: &str) -> bool {
    matches!(name, "x" | "y" | "z")
}
