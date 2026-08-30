//! CSV read/write (ADR-0004): header-driven, `x`/`y`/`z` required; standard
//! names adopt their pinned dtype, everything else round-trips as f8.

use crate::cloud::dims::{standard_dim_dtype, DType, DimArray};
use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use crate::io::ColumnMap;

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
/// direct lookup; ADR-0004: the same map applies on read and write).
fn file_col_name(columns: Option<&ColumnMap>, dim: &str) -> String {
    columns
        .and_then(|m| m.get(dim))
        .cloned()
        .unwrap_or_else(|| dim.to_string())
}

fn find_required(headers: &[String], name: &str) -> Result<usize> {
    headers
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| Error::value(format!("CSV is missing a required '{name}' column")))
}

fn parse_f64(record: &csv::StringRecord, idx: usize, name: &str) -> Result<f64> {
    record
        .get(idx)
        .ok_or_else(|| Error::value(format!("CSV row is missing column '{name}'")))?
        .trim()
        .parse::<f64>()
        .map_err(|e| Error::value(format!("CSV column '{name}' is not numeric: {e}")))
}

/// Casts a parsed f64 column to its pinned dtype (standard names) or leaves
/// it as f8 (unknown names become f8 extra dims, ADR-0004).
fn cast_f64_column(name: &str, values: Vec<f64>) -> Result<DimArray> {
    let Some(dtype) = standard_dim_dtype(name) else {
        return Ok(DimArray::F64(values));
    };
    Ok(match dtype {
        DType::U8 => DimArray::U8(round_checked(
            name,
            values,
            u8::MIN as f64,
            u8::MAX as f64,
            |v| v as u8,
        )?),
        DType::U16 => DimArray::U16(round_checked(
            name,
            values,
            u16::MIN as f64,
            u16::MAX as f64,
            |v| v as u16,
        )?),
        DType::I16 => DimArray::I16(round_checked(
            name,
            values,
            i16::MIN as f64,
            i16::MAX as f64,
            |v| v as i16,
        )?),
        DType::F64 => DimArray::F64(values),
        DType::Bool => DimArray::Bool(values.into_iter().map(|v| v != 0.0).collect()),
        DType::U32 | DType::U64 | DType::I8 | DType::I32 | DType::I64 | DType::F32 => {
            unreachable!("the standard dimension table only uses u8/u16/i16/f64/bool")
        }
    })
}

fn round_checked<T>(
    name: &str,
    values: Vec<f64>,
    min: f64,
    max: f64,
    cast: impl Fn(f64) -> T,
) -> Result<Vec<T>> {
    values
        .into_iter()
        .map(|v| {
            let rounded = v.round();
            if rounded < min || rounded > max {
                Err(Error::overflow(format!(
                    "CSV column '{name}' value {v} does not fit its dimension's dtype range [{min}, {max}]"
                )))
            } else {
                Ok(cast(rounded))
            }
        })
        .collect()
}

pub fn read(path: &str, columns: Option<&ColumnMap>, delimiter: u8) -> Result<PointCloud> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_path(path)?;

    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| mapped_name(columns, h))
        .collect();

    let x_idx = find_required(&headers, "x")?;
    let y_idx = find_required(&headers, "y")?;
    let z_idx = find_required(&headers, "z")?;
    let other_cols: Vec<(usize, String)> = headers
        .iter()
        .enumerate()
        .filter(|(i, _)| ![x_idx, y_idx, z_idx].contains(i))
        .map(|(i, name)| (i, name.clone()))
        .collect();

    let mut xyz = Vec::new();
    let mut other_values: Vec<Vec<f64>> = vec![Vec::new(); other_cols.len()];

    for record in reader.records() {
        let record = record?;
        xyz.push(parse_f64(&record, x_idx, "x")?);
        xyz.push(parse_f64(&record, y_idx, "y")?);
        xyz.push(parse_f64(&record, z_idx, "z")?);
        for (slot, (idx, name)) in other_cols.iter().enumerate() {
            other_values[slot].push(parse_f64(&record, *idx, name)?);
        }
    }

    let mut cloud = PointCloud::new();
    cloud.set_coords_f64(&xyz)?;
    for ((_, name), values) in other_cols.into_iter().zip(other_values) {
        cloud.set_dim_array(&name, cast_f64_column(&name, values)?)?;
    }
    Ok(cloud)
}

fn format_dim_value(cloud: &PointCloud, name: &str, xyz: &[f64], i: usize) -> String {
    match name {
        "x" => xyz[i * 3].to_string(),
        "y" => xyz[i * 3 + 1].to_string(),
        "z" => xyz[i * 3 + 2].to_string(),
        _ => match cloud
            .dim(name)
            .expect("dim_names() only returns existing dimensions")
        {
            DimArray::U8(v) => v[i].to_string(),
            DimArray::U16(v) => v[i].to_string(),
            DimArray::U32(v) => v[i].to_string(),
            DimArray::U64(v) => v[i].to_string(),
            DimArray::I8(v) => v[i].to_string(),
            DimArray::I16(v) => v[i].to_string(),
            DimArray::I32(v) => v[i].to_string(),
            DimArray::I64(v) => v[i].to_string(),
            DimArray::F32(v) => v[i].to_string(),
            DimArray::F64(v) => v[i].to_string(),
            DimArray::Bool(v) => if v[i] { "1" } else { "0" }.to_string(),
        },
    }
}

pub fn write(
    cloud: &PointCloud,
    path: &str,
    columns: Option<&ColumnMap>,
    delimiter: u8,
) -> Result<()> {
    if !cloud.has_dim("x") {
        return Err(Error::value("point cloud has no coordinates to write"));
    }
    let mut writer = csv::WriterBuilder::new()
        .delimiter(delimiter)
        .from_path(path)?;

    let dim_names = cloud.dim_names();
    let header: Vec<String> = dim_names
        .iter()
        .map(|n| file_col_name(columns, n))
        .collect();
    writer.write_record(&header)?;

    // Zero-copy for CPU-f64 clouds; materializes only when the
    // representation forces it.
    let xyz = cloud
        .coords_f64_cow()
        .expect("has_dim(\"x\") implies coordinates exist");
    for i in 0..cloud.len() {
        let row: Vec<String> = dim_names
            .iter()
            .map(|name| format_dim_value(cloud, name, &xyz, i))
            .collect();
        writer.write_record(&row)?;
    }
    writer.flush()?;
    Ok(())
}
