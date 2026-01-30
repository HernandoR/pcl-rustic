/// LAZ/LAS格式读写/删，处理压缩/解压缩、格式兼容
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::{IOConvert, PointCloudCore, PointCloudProperties};
use crate::utils::error::Result;
use las::{Reader, Writer, Builder, Point, Color};
use las::point::Format;
use las::Write;
use std::fs;

impl HighPerformancePointCloud {
    /// 从LAS/LAZ文件读取（自动检测压缩）
    pub fn from_las_laz(path: &str) -> Result<Self> {
        if !fs::metadata(path).is_ok() {
            return Err(format!("文件不存在: {}", path).into());
        }

        let file = fs::File::open(path).map_err(|e| e.into())?;
        let mut reader = Reader::new(file).map_err(|_| "LAS文件格式错误".to_string())?;

        let mut xyz = Vec::new();
        let mut intensity = Vec::new();
        let mut rgb = Vec::new();
        let has_intensity = reader.header().point_format().has_intensity;
        let has_rgb = reader.header().point_format().has_rgb;

        for point_result in reader.points() {
            let point = point_result.map_err(|_| "读取LAS点失败".to_string())?;

            xyz.push(vec![point.x as f32, point.y as f32, point.z as f32]);

            if has_intensity {
                intensity.push(point.intensity as f32 / 65535.0);
            }

            if has_rgb {
                if let Some(color) = point.color {
                    rgb.push(vec![
                        (color.red >> 8) as u8,
                        (color.green >> 8) as u8,
                        (color.blue >> 8) as u8,
                    ]);
                }
            }
        }

        let mut result = Self::from_xyz(xyz)?;

        if !intensity.is_empty() && intensity.len() == result.point_count() {
            result.set_intensity(intensity)?;
        }

        if !rgb.is_empty() && rgb.len() == result.point_count() {
            result.set_rgb(rgb)?;
        }

        Ok(result)
    }

    /// 写入LAS文件
    /// compress: 是否压缩为LAZ格式
    pub fn to_las(&self, path: &str, _compress: bool) -> Result<()> {
        if self.point_count() == 0 {
            return Err("点云为空".into());
        }

        // 验证路径可写
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(|e| e.into())?;
            }
        }

        let has_rgb = self.has_rgb();
        let has_intensity = self.has_intensity();
        let format_id = if has_rgb { 2 } else { 0 };

        let mut builder = Builder::from((1, 4));
        let mut format = Format::new(format_id).map_err(|e| e.to_string())?;
        format.is_compressed = _compress || path.to_lowercase().ends_with(".laz");
        builder.point_format = format;
        let header = builder.into_header().map_err(|e| e.to_string())?;

        let mut writer = Writer::from_path(path, header).map_err(|e| e.to_string())?;

        let xyz = self.get_xyz();
        let intensity = self.get_intensity();
        let rgb = self.get_rgb();

        for (idx, point_xyz) in xyz.iter().enumerate() {
            let mut point = Point::default();
            point.x = point_xyz[0] as f64;
            point.y = point_xyz[1] as f64;
            point.z = point_xyz[2] as f64;

            if let Some(intensity_vec) = &intensity {
                let raw = (intensity_vec[idx] * 65535.0).clamp(0.0, 65535.0);
                point.intensity = raw as u16;
            } else if !has_intensity {
                point.intensity = 0;
            }

            if let Some(rgb_vec) = &rgb {
                let c = &rgb_vec[idx];
                point.color = Some(Color {
                    red: (c[0] as u16) << 8,
                    green: (c[1] as u16) << 8,
                    blue: (c[2] as u16) << 8,
                });
            }

            writer.write(point).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// 删除LAS/LAZ文件
    pub fn delete_file(path: &str) -> Result<()> {
        if !fs::metadata(path).is_ok() {
            return Err(format!("文件不存在: {}", path).into());
        }

        fs::remove_file(path).map_err(|e| e.into())?;
        Ok(())
    }
}

impl IOConvert for HighPerformancePointCloud {
    fn from_las_laz(path: &str) -> Result<Self>
    where
        Self: Sized,
    {
        Self::from_las_laz(path)
    }

    fn from_parquet(path: &str) -> Result<Self>
    where
        Self: Sized,
    {
        crate::io::parquet::from_parquet(path)
    }

    fn from_csv(path: &str, delimiter: u8) -> Result<Self>
    where
        Self: Sized,
    {
        crate::io::csv::from_csv(path, delimiter)
    }

    fn to_las(&self, path: &str, compress: bool) -> Result<()> {
        self.to_las(path, compress)
    }

    fn to_parquet(&self, path: &str) -> Result<()> {
        crate::io::parquet::to_parquet(self, path)
    }

    fn to_csv(&self, path: &str, delimiter: u8) -> Result<()> {
        crate::io::csv::to_csv(self, path, delimiter)
    }

    fn delete_file(path: &str) -> Result<()>
    where
        Self: Sized,
    {
        HighPerformancePointCloud::delete_file(path)
    }
}
