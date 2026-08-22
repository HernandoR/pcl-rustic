//! LAS/LAZ read/write via the `las` crate (ADR-0004, ADR-0002).
//!
//! Known `las` 0.9 crate limitations, recorded here rather than worked
//! around:
//!
//! - `scan_angle`: the crate always exposes `Point::scan_angle` as `f32`
//!   degrees (`las-0.9.11/src/point/mod.rs`), discarding whether the source
//!   was a 1-degree `scan_angle_rank` (formats 0-5) or a 0.006-degree-scaled
//!   `scan_angle` (formats 6-10; `las-0.9.11/src/raw/point.rs`). We
//!   reconstruct our i16 dimension from that f32 using the same 0.006 scale
//!   factor the crate uses internally for extended formats (mirrored below
//!   as `SCAN_ANGLE_SCALE`, which the crate does not export), and a direct
//!   whole-degree widen for non-extended formats -- matching ADR-0002's
//!   "widens losslessly for 0-5, native scale for 6-10" rule as closely as
//!   the crate's public API allows.
//! - Extra dimensions are NOT written as LAS ExtraBytes: the crate only
//!   exposes `Point::extra_bytes: Vec<u8>`, an undifferentiated raw blob
//!   with no VLR name/dtype metadata (no `ExtraBytesParams`-equivalent
//!   exists in this crate's public API). There is no way to declare named,
//!   typed extra dimensions through it, so LAS write simply omits extra
//!   dimensions; Parquet is the lossless round-trip carrier for them
//!   (ADR-0004).

use crate::cloud::dims::DimArray;
use crate::cloud::PointCloud;
use crate::error::{Error, Result};
use las::point::{Classification, Format, ScanDirection};
use las::{Color, Point, Reader, Writer};

/// LAS 1.4's scan-angle scale factor (`las-0.9.11/src/raw/point.rs`,
/// `SCAN_ANGLE_SCALE_FACTOR`; not exported by the crate).
const SCAN_ANGLE_SCALE: f32 = 0.006;

pub fn read(path: &str) -> Result<PointCloud> {
    let mut reader = Reader::from_path(path)?;
    let header = reader.header().clone();
    let format = *header.point_format();
    let transforms = *header.transforms();
    let n = header.number_of_points() as usize;

    let mut xyz = Vec::with_capacity(n * 3);
    let mut intensity = Vec::with_capacity(n);
    let mut return_number = Vec::with_capacity(n);
    let mut number_of_returns = Vec::with_capacity(n);
    let mut scan_direction_flag = Vec::with_capacity(n);
    let mut edge_of_flight_line = Vec::with_capacity(n);
    let mut classification = Vec::with_capacity(n);
    let mut synthetic = Vec::with_capacity(n);
    let mut key_point = Vec::with_capacity(n);
    let mut withheld = Vec::with_capacity(n);
    let mut overlap = Vec::with_capacity(n);
    let mut scanner_channel = Vec::with_capacity(n);
    let mut scan_angle = Vec::with_capacity(n);
    let mut user_data = Vec::with_capacity(n);
    let mut point_source_id = Vec::with_capacity(n);
    let mut gps_time: Vec<f64> = Vec::with_capacity(if format.has_gps_time { n } else { 0 });
    let mut red: Vec<u16> = Vec::with_capacity(if format.has_color { n } else { 0 });
    let mut green: Vec<u16> = Vec::with_capacity(if format.has_color { n } else { 0 });
    let mut blue: Vec<u16> = Vec::with_capacity(if format.has_color { n } else { 0 });
    let mut nir: Vec<u16> = Vec::with_capacity(if format.has_nir { n } else { 0 });

    for point in reader.points() {
        let point = point?;
        xyz.push(point.x);
        xyz.push(point.y);
        xyz.push(point.z);
        intensity.push(point.intensity);
        return_number.push(point.return_number);
        number_of_returns.push(point.number_of_returns);
        scan_direction_flag.push(matches!(point.scan_direction, ScanDirection::LeftToRight));
        edge_of_flight_line.push(point.is_edge_of_flight_line);
        classification.push(u8::from(point.classification));
        synthetic.push(point.is_synthetic);
        key_point.push(point.is_key_point);
        withheld.push(point.is_withheld);
        overlap.push(point.is_overlap);
        if format.is_extended {
            scanner_channel.push(point.scanner_channel);
            scan_angle.push((point.scan_angle / SCAN_ANGLE_SCALE).round() as i16);
        } else {
            scan_angle.push(point.scan_angle.round() as i16);
        }
        user_data.push(point.user_data);
        point_source_id.push(point.point_source_id);
        if format.has_gps_time {
            gps_time.push(point.gps_time.unwrap_or(0.0));
        }
        if format.has_color {
            let color = point.color.unwrap_or_default();
            red.push(color.red);
            green.push(color.green);
            blue.push(color.blue);
        }
        if format.has_nir {
            nir.push(point.nir.unwrap_or(0));
        }
    }

    let mut cloud = PointCloud::new();
    cloud.set_scales([transforms.x.scale, transforms.y.scale, transforms.z.scale]);
    cloud.set_coords_f64_with_offset(
        &xyz,
        [
            transforms.x.offset,
            transforms.y.offset,
            transforms.z.offset,
        ],
    )?;

    cloud.set_dim_array("intensity", DimArray::U16(intensity))?;
    cloud.set_dim_array("return_number", DimArray::U8(return_number))?;
    cloud.set_dim_array("number_of_returns", DimArray::U8(number_of_returns))?;
    cloud.set_dim_array("scan_direction_flag", DimArray::Bool(scan_direction_flag))?;
    cloud.set_dim_array("edge_of_flight_line", DimArray::Bool(edge_of_flight_line))?;
    cloud.set_dim_array("classification", DimArray::U8(classification))?;
    cloud.set_dim_array("synthetic", DimArray::Bool(synthetic))?;
    cloud.set_dim_array("key_point", DimArray::Bool(key_point))?;
    cloud.set_dim_array("withheld", DimArray::Bool(withheld))?;
    cloud.set_dim_array("overlap", DimArray::Bool(overlap))?;
    cloud.set_dim_array("scan_angle", DimArray::I16(scan_angle))?;
    cloud.set_dim_array("user_data", DimArray::U8(user_data))?;
    cloud.set_dim_array("point_source_id", DimArray::U16(point_source_id))?;
    if format.is_extended {
        cloud.set_dim_array("scanner_channel", DimArray::U8(scanner_channel))?;
    }
    if format.has_gps_time {
        cloud.set_dim_array("gps_time", DimArray::F64(gps_time))?;
    }
    if format.has_color {
        cloud.set_dim_array("red", DimArray::U16(red))?;
        cloud.set_dim_array("green", DimArray::U16(green))?;
        cloud.set_dim_array("blue", DimArray::U16(blue))?;
    }
    if format.has_nir {
        cloud.set_dim_array("nir", DimArray::U16(nir))?;
    }

    Ok(cloud)
}

/// Picks the minimal LAS point format that carries every present dimension
/// (ADR-0004): NIR or `scanner_channel` force an extended format (6-10);
/// otherwise gps_time and/or RGB select among 0-3.
fn select_format(cloud: &PointCloud) -> Result<Format> {
    let has_gps_time = cloud.has_dim("gps_time");
    let has_rgb = cloud.has_dim("red") && cloud.has_dim("green") && cloud.has_dim("blue");
    let has_nir = cloud.has_dim("nir");
    let has_scanner_channel = cloud.has_dim("scanner_channel");

    let id = if has_nir || has_scanner_channel {
        if has_rgb && has_nir {
            8
        } else if has_rgb {
            7
        } else {
            6
        }
    } else if has_gps_time && has_rgb {
        3
    } else if has_gps_time {
        1
    } else if has_rgb {
        2
    } else {
        0
    };
    Format::new(id).map_err(Error::from)
}

fn get_u8(cloud: &PointCloud, name: &str, i: usize) -> u8 {
    match cloud.dim(name) {
        Some(DimArray::U8(v)) => v[i],
        _ => 0,
    }
}

fn get_u16(cloud: &PointCloud, name: &str, i: usize) -> u16 {
    match cloud.dim(name) {
        Some(DimArray::U16(v)) => v[i],
        _ => 0,
    }
}

fn get_i16(cloud: &PointCloud, name: &str, i: usize) -> i16 {
    match cloud.dim(name) {
        Some(DimArray::I16(v)) => v[i],
        _ => 0,
    }
}

fn get_bool(cloud: &PointCloud, name: &str, i: usize) -> bool {
    match cloud.dim(name) {
        Some(DimArray::Bool(v)) => v[i],
        _ => false,
    }
}

fn get_f64(cloud: &PointCloud, name: &str, i: usize) -> f64 {
    match cloud.dim(name) {
        Some(DimArray::F64(v)) => v[i],
        _ => 0.0,
    }
}

pub fn write(cloud: &PointCloud, path: &str) -> Result<()> {
    let n = cloud.len();
    let coords = cloud
        .coords_f64()
        .ok_or_else(|| Error::value("point cloud has no coordinates to write"))?;

    let format = select_format(cloud)?;
    let scales = cloud.scales();
    let offset = cloud.offset();

    let mut builder = las::Builder::from((1, 4));
    builder.point_format = format;
    builder.transforms = las::Vector {
        x: las::Transform {
            scale: scales[0],
            offset: offset[0],
        },
        y: las::Transform {
            scale: scales[1],
            offset: offset[1],
        },
        z: las::Transform {
            scale: scales[2],
            offset: offset[2],
        },
    };
    let header = builder.into_header().map_err(Error::from)?;

    // `Writer::from_path` picks LAS vs. LAZ from `path`'s extension itself.
    let mut writer = Writer::from_path(path, header)?;

    for i in 0..n {
        let point = Point {
            x: coords[i * 3],
            y: coords[i * 3 + 1],
            z: coords[i * 3 + 2],
            intensity: get_u16(cloud, "intensity", i),
            return_number: get_u8(cloud, "return_number", i),
            number_of_returns: get_u8(cloud, "number_of_returns", i),
            scan_direction: if get_bool(cloud, "scan_direction_flag", i) {
                ScanDirection::LeftToRight
            } else {
                ScanDirection::RightToLeft
            },
            is_edge_of_flight_line: get_bool(cloud, "edge_of_flight_line", i),
            classification: Classification::new(get_u8(cloud, "classification", i))
                .map_err(|e| Error::value(format!("invalid classification code: {e}")))?,
            is_synthetic: get_bool(cloud, "synthetic", i),
            is_key_point: get_bool(cloud, "key_point", i),
            is_withheld: get_bool(cloud, "withheld", i),
            is_overlap: get_bool(cloud, "overlap", i),
            scanner_channel: if format.is_extended {
                get_u8(cloud, "scanner_channel", i)
            } else {
                0
            },
            scan_angle: if format.is_extended {
                get_i16(cloud, "scan_angle", i) as f32 * SCAN_ANGLE_SCALE
            } else {
                get_i16(cloud, "scan_angle", i) as f32
            },
            user_data: get_u8(cloud, "user_data", i),
            point_source_id: get_u16(cloud, "point_source_id", i),
            gps_time: format.has_gps_time.then(|| get_f64(cloud, "gps_time", i)),
            color: format.has_color.then(|| {
                Color::new(
                    get_u16(cloud, "red", i),
                    get_u16(cloud, "green", i),
                    get_u16(cloud, "blue", i),
                )
            }),
            waveform: None,
            nir: format.has_nir.then(|| get_u16(cloud, "nir", i)),
            extra_bytes: Vec::new(),
        };
        writer.write_point(point)?;
    }
    writer.close()?;
    Ok(())
}
