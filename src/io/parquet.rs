//! Parquet read/write via `arrow`/`parquet` (ADR-0004): one column per
//! dimension, with an exact Arrow-type <-> dtype mapping both ways. polars
//! is dropped; record batches map 1:1 onto the typed column store.

use crate::cloud::dims::{standard_dim_dtype, DType, DimArray};
use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use crate::io::ColumnMap;
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, RecordBatch, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::sync::Arc;

/// Maps a file column name to a dimension name (`columns`: dim -> file, so
/// reading inverts the map).
fn mapped_name(columns: Option<&ColumnMap>, file_col: &str) -> String {
    columns
        .and_then(|m| {
            m.iter()
                .find(|(_, v)| v.as_str() == file_col)
                .map(|(k, _)| k.clone())
        })
        .unwrap_or_else(|| file_col.to_string())
}

/// Maps a dimension name to a file column name (`columns`: dim -> file,
/// direct lookup).
fn file_col_name(columns: Option<&ColumnMap>, dim: &str) -> String {
    columns
        .and_then(|m| m.get(dim))
        .cloned()
        .unwrap_or_else(|| dim.to_string())
}

fn arrow_type_of(dtype: DType) -> DataType {
    match dtype {
        DType::U8 => DataType::UInt8,
        DType::U16 => DataType::UInt16,
        DType::U32 => DataType::UInt32,
        DType::U64 => DataType::UInt64,
        DType::I8 => DataType::Int8,
        DType::I16 => DataType::Int16,
        DType::I32 => DataType::Int32,
        DType::I64 => DataType::Int64,
        DType::F32 => DataType::Float32,
        DType::F64 => DataType::Float64,
        DType::Bool => DataType::Boolean,
    }
}

fn dtype_of_arrow(data_type: &DataType) -> Option<DType> {
    Some(match data_type {
        DataType::UInt8 => DType::U8,
        DataType::UInt16 => DType::U16,
        DataType::UInt32 => DType::U32,
        DataType::UInt64 => DType::U64,
        DataType::Int8 => DType::I8,
        DataType::Int16 => DType::I16,
        DataType::Int32 => DType::I32,
        DataType::Int64 => DType::I64,
        DataType::Float32 => DType::F32,
        DataType::Float64 => DType::F64,
        DataType::Boolean => DType::Bool,
        _ => return None,
    })
}

fn dim_array_to_arrow(dim: &DimArray) -> ArrayRef {
    match dim {
        DimArray::U8(v) => Arc::new(UInt8Array::from(v.clone())),
        DimArray::U16(v) => Arc::new(UInt16Array::from(v.clone())),
        DimArray::U32(v) => Arc::new(UInt32Array::from(v.clone())),
        DimArray::U64(v) => Arc::new(UInt64Array::from(v.clone())),
        DimArray::I8(v) => Arc::new(Int8Array::from(v.clone())),
        DimArray::I16(v) => Arc::new(Int16Array::from(v.clone())),
        DimArray::I32(v) => Arc::new(Int32Array::from(v.clone())),
        DimArray::I64(v) => Arc::new(Int64Array::from(v.clone())),
        DimArray::F32(v) => Arc::new(Float32Array::from(v.clone())),
        DimArray::F64(v) => Arc::new(Float64Array::from(v.clone())),
        DimArray::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
    }
}

pub fn write(cloud: &PointCloud, path: &str, columns: Option<&ColumnMap>) -> Result<()> {
    if !cloud.has_dim("x") {
        return Err(Error::value("point cloud has no coordinates to write"));
    }
    let n = cloud.len();
    // Zero-copy for CPU-f64 clouds; materializes only when the
    // representation forces it.
    let xyz = cloud
        .coords_f64_cow()
        .expect("has_dim(\"x\") implies coordinates exist");

    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for name in cloud.dim_names() {
        let file_col = file_col_name(columns, &name);
        let (data_type, array): (DataType, ArrayRef) = match name.as_str() {
            "x" => (
                DataType::Float64,
                Arc::new(Float64Array::from(
                    (0..n).map(|i| xyz[i * 3]).collect::<Vec<_>>(),
                )),
            ),
            "y" => (
                DataType::Float64,
                Arc::new(Float64Array::from(
                    (0..n).map(|i| xyz[i * 3 + 1]).collect::<Vec<_>>(),
                )),
            ),
            "z" => (
                DataType::Float64,
                Arc::new(Float64Array::from(
                    (0..n).map(|i| xyz[i * 3 + 2]).collect::<Vec<_>>(),
                )),
            ),
            _ => {
                let dim = cloud
                    .dim(&name)
                    .expect("dim_names() only returns existing dimensions");
                (arrow_type_of(dim.dtype()), dim_array_to_arrow(dim))
            }
        };
        fields.push(Field::new(file_col, data_type, false));
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn resolve_column_dtype(name: &str, arrow_type: &DataType) -> Result<DType> {
    let mapped = dtype_of_arrow(arrow_type).ok_or_else(|| {
        Error::value(format!(
            "parquet column '{name}' has unsupported Arrow type {arrow_type:?}"
        ))
    })?;
    if let Some(standard) = standard_dim_dtype(name) {
        if mapped != standard {
            return Err(Error::value(format!(
                "parquet column '{name}' has Arrow type {arrow_type:?} ({}), but the standard dimension '{name}' requires {}",
                mapped.numpy_name(),
                standard.numpy_name()
            )));
        }
        Ok(standard)
    } else if mapped == DType::Bool {
        Err(Error::value(format!(
            "parquet column '{name}' is boolean, which is not an allowed extra-dimension dtype"
        )))
    } else {
        Ok(mapped)
    }
}

fn extract_f64_column(batches: &[RecordBatch], col_idx: usize, name: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for batch in batches {
        let array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::value(format!("parquet column '{name}' must be float64")))?;
        if array.null_count() > 0 {
            return Err(Error::value(format!("parquet column '{name}' contains nulls, which pcl-rustic dimensions do not support")));
        }
        out.extend(array.values().iter().copied());
    }
    Ok(out)
}

macro_rules! extract_primitive {
    ($batches:expr, $col_idx:expr, $name:expr, $arrow_ty:ty, $variant:ident) => {{
        let mut v = Vec::new();
        for batch in $batches {
            let array = batch
                .column($col_idx)
                .as_any()
                .downcast_ref::<$arrow_ty>()
                .ok_or_else(|| Error::value(format!("parquet column '{}' has inconsistent type across row groups", $name)))?;
            if array.null_count() > 0 {
                return Err(Error::value(format!("parquet column '{}' contains nulls, which pcl-rustic dimensions do not support", $name)));
            }
            v.extend(array.values().iter().copied());
        }
        DimArray::$variant(v)
    }};
}

fn extract_column(
    batches: &[RecordBatch],
    col_idx: usize,
    dtype: DType,
    name: &str,
) -> Result<DimArray> {
    Ok(match dtype {
        DType::U8 => extract_primitive!(batches, col_idx, name, UInt8Array, U8),
        DType::U16 => extract_primitive!(batches, col_idx, name, UInt16Array, U16),
        DType::U32 => extract_primitive!(batches, col_idx, name, UInt32Array, U32),
        DType::U64 => extract_primitive!(batches, col_idx, name, UInt64Array, U64),
        DType::I8 => extract_primitive!(batches, col_idx, name, Int8Array, I8),
        DType::I16 => extract_primitive!(batches, col_idx, name, Int16Array, I16),
        DType::I32 => extract_primitive!(batches, col_idx, name, Int32Array, I32),
        DType::I64 => extract_primitive!(batches, col_idx, name, Int64Array, I64),
        DType::F32 => extract_primitive!(batches, col_idx, name, Float32Array, F32),
        DType::F64 => extract_primitive!(batches, col_idx, name, Float64Array, F64),
        DType::Bool => {
            let mut v = Vec::new();
            for batch in batches {
                let array = batch
                    .column(col_idx)
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        Error::value(format!(
                            "parquet column '{name}' has inconsistent type across row groups"
                        ))
                    })?;
                if array.null_count() > 0 {
                    return Err(Error::value(format!("parquet column '{name}' contains nulls, which pcl-rustic dimensions do not support")));
                }
                v.extend(
                    array
                        .iter()
                        .map(|b| b.expect("null_count() == 0 checked above")),
                );
            }
            DimArray::Bool(v)
        }
    })
}

pub fn read(path: &str, columns: Option<&ColumnMap>) -> Result<PointCloud> {
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let dim_names: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| mapped_name(columns, f.name()))
        .collect();

    let x_idx = dim_names
        .iter()
        .position(|n| n == "x")
        .ok_or_else(|| Error::value("parquet file is missing a required 'x' column"))?;
    let y_idx = dim_names
        .iter()
        .position(|n| n == "y")
        .ok_or_else(|| Error::value("parquet file is missing a required 'y' column"))?;
    let z_idx = dim_names
        .iter()
        .position(|n| n == "z")
        .ok_or_else(|| Error::value("parquet file is missing a required 'z' column"))?;

    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch?);
    }

    let x = extract_f64_column(&batches, x_idx, "x")?;
    let y = extract_f64_column(&batches, y_idx, "y")?;
    let z = extract_f64_column(&batches, z_idx, "z")?;
    let n = x.len();
    let mut xyz = Vec::with_capacity(n * 3);
    for i in 0..n {
        xyz.push(x[i]);
        xyz.push(y[i]);
        xyz.push(z[i]);
    }

    let mut cloud = PointCloud::new();
    cloud.set_coords_f64(&xyz)?;

    for (idx, name) in dim_names.iter().enumerate() {
        if idx == x_idx || idx == y_idx || idx == z_idx {
            continue;
        }
        let dtype = resolve_column_dtype(name, schema.field(idx).data_type())?;
        let dim = extract_column(&batches, idx, dtype, name)?;
        cloud.set_dim_array(name, dim)?;
    }

    Ok(cloud)
}
