//! Extension-based file format dispatch (ADR-0004): `.las`/`.laz`/`.csv`/
//! `.parquet`, mirroring laspy's `read`/`write` entry points.

mod csv;
mod las;
mod parquet;

use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// `dimension name -> file column name` mapping used by CSV/Parquet I/O.
/// Reads invert the same map (file column name -> dimension name).
pub type ColumnMap = HashMap<String, String>;

fn extension_of(path: &str) -> Result<String> {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .ok_or_else(|| {
            Error::os(format!(
                "cannot determine file format from path '{path}' (no extension)"
            ))
        })
}

impl PointCloud {
    /// Reads a point cloud, dispatching on `path`'s extension.
    pub fn read_file(path: &str, columns: Option<&ColumnMap>, delimiter: u8) -> Result<Self> {
        match extension_of(path)?.as_str() {
            "las" | "laz" => las::read(path),
            "csv" | "tsv" => csv::read(path, columns, delimiter),
            "parquet" => parquet::read(path, columns),
            other => Err(Error::os(format!("unsupported file format '.{other}'"))),
        }
    }

    /// Writes this point cloud, dispatching on `path`'s extension. `.laz`
    /// implies compression.
    pub fn write_file(&self, path: &str, columns: Option<&ColumnMap>, delimiter: u8) -> Result<()> {
        match extension_of(path)?.as_str() {
            "las" | "laz" => las::write(self, path),
            "csv" | "tsv" => csv::write(self, path, columns, delimiter),
            "parquet" => parquet::write(self, path, columns),
            other => Err(Error::os(format!("unsupported file format '.{other}'"))),
        }
    }
}
