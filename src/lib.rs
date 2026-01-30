mod traits;
mod utils;
mod point_cloud;
mod io;
mod interop;

use pyo3::prelude::*;
use point_cloud::core::HighPerformancePointCloud;
use traits::{PointCloudCore, PointCloudProperties, CoordinateTransform, VoxelDownsample, DownsampleStrategy};

/// Python模块入口
#[pymodule]
fn _core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyDownsampleStrategy>()?;
    Ok(())
}

/// Python级别的点云类
#[pyclass(name = "PointCloud")]
pub struct PyPointCloud {
    inner: HighPerformancePointCloud,
}

#[pymethods]
impl PyPointCloud {
    /// 创建空点云
    #[new]
    fn new() -> Self {
        PyPointCloud {
            inner: HighPerformancePointCloud::new(),
        }
    }

    /// 从XYZ坐标创建点云
    #[staticmethod]
    fn from_xyz(xyz: Vec<Vec<f32>>) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_xyz(xyz)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner })
    }

    /// 获取点数
    fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    /// 获取XYZ坐标
    fn get_xyz(&self) -> Vec<Vec<f32>> {
        self.inner.get_xyz()
    }

    /// 检查是否有intensity
    fn has_intensity(&self) -> bool {
        self.inner.has_intensity()
    }

    /// 检查是否有RGB
    fn has_rgb(&self) -> bool {
        self.inner.has_rgb()
    }

    /// 获取intensity
    fn get_intensity(&self) -> Option<Vec<f32>> {
        self.inner.get_intensity()
    }

    /// 获取RGB
    fn get_rgb(&self) -> Option<Vec<Vec<u8>>> {
        self.inner.get_rgb()
    }

    /// 设置intensity
    fn set_intensity(&mut self, intensity: Vec<f32>) -> PyResult<()> {
        self.inner.set_intensity(intensity)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 设置RGB
    fn set_rgb(&mut self, rgb: Vec<Vec<u8>>) -> PyResult<()> {
        self.inner.set_rgb(rgb)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 添加自定义属性
    fn add_attribute(&mut self, name: String, data: Vec<f32>) -> PyResult<()> {
        self.inner.add_attribute(name, data)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 设置自定义属性
    fn set_attribute(&mut self, name: String, data: Vec<f32>) -> PyResult<()> {
        self.inner.set_attribute(name, data)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 获取属性名列表
    fn attribute_names(&self) -> Vec<String> {
        self.inner.attribute_names()
    }

    /// 获取属性
    fn get_attribute(&self, name: &str) -> Option<Vec<f32>> {
        self.inner.get_attribute(name)
    }

    /// 删除属性
    fn remove_attribute(&mut self, name: &str) -> PyResult<()> {
        self.inner.remove_attribute(name)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 坐标变换（矩阵）
    fn transform(&self, matrix: Vec<Vec<f32>>) -> PyResult<Self> {
        let result = self.inner.transform(matrix)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner: result })
    }

    /// 刚体变换（旋转+平移）
    fn rigid_transform(
        &self,
        rotation: Vec<Vec<f32>>,
        translation: Vec<f32>,
    ) -> PyResult<Self> {
        let result = self.inner.rigid_transform(rotation, translation)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner: result })
    }

    /// 体素下采样
    fn voxel_downsample(&self, voxel_size: f32, strategy: i32) -> PyResult<Self> {
        let strategy_impl: Box<dyn DownsampleStrategy> = match strategy {
            0 => Box::new(point_cloud::voxel::RandomSampleStrategy),
            1 => Box::new(point_cloud::voxel::CentroidSampleStrategy),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                "未知的采样策略"
            )),
        };

        let result = self.inner.voxel_downsample(voxel_size, strategy_impl)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner: result })
    }

    /// 从LAS/LAZ文件读取
    #[staticmethod]
    fn from_las(path: &str) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_las_laz(path)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner })
    }

    /// 保存为LAS文件
    fn to_las(&self, path: &str, compress: bool) -> PyResult<()> {
        self.inner.to_las(path, compress)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 从CSV读取
    #[staticmethod]
    #[pyo3(signature = (
        path,
        delimiter = b',',
        x = None,
        y = None,
        z = None,
        intensity = None,
        rgb_r = None,
        rgb_g = None,
        rgb_b = None
    ))]
    fn from_csv(
        path: &str,
        delimiter: u8,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<Self> {
        let columns = io::table::TableColumns::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        let inner = HighPerformancePointCloud::from_table_csv(path, delimiter, columns)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner })
    }

    /// 从Parquet读取
    #[staticmethod]
    #[pyo3(signature = (
        path,
        x = None,
        y = None,
        z = None,
        intensity = None,
        rgb_r = None,
        rgb_g = None,
        rgb_b = None
    ))]
    fn from_parquet(
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<Self> {
        let columns = io::table::TableColumns::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        let inner = HighPerformancePointCloud::from_table_parquet(path, columns)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner })
    }

    /// 从Parquet读取
    #[staticmethod]
    fn from_parquet(path: &str) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_parquet(path)
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner })
    }

    /// 保存为CSV
    #[pyo3(signature = (
        path,
        delimiter = b',',
        x = None,
        y = None,
        z = None,
        intensity = None,
        rgb_r = None,
        rgb_g = None,
        rgb_b = None
    ))]
    fn to_csv(
        &self,
        path: &str,
        delimiter: u8,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<()> {
        let columns = io::table::TableColumns::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        self.inner.to_table_csv(path, delimiter, columns)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 保存为Parquet
    #[pyo3(signature = (
        path,
        x = None,
        y = None,
        z = None,
        intensity = None,
        rgb_r = None,
        rgb_g = None,
        rgb_b = None
    ))]
    fn to_parquet(
        &self,
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<()> {
        let columns = io::table::TableColumns::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        self.inner.to_table_parquet(path, columns)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 保存为Parquet
    fn to_parquet(&self, path: &str) -> PyResult<()> {
        self.inner.to_parquet(path)
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 根据扩展名自动读取
    #[staticmethod]
    #[pyo3(signature = (
        path,
        x = None,
        y = None,
        z = None,
        intensity = None,
        rgb_r = None,
        rgb_g = None,
        rgb_b = None
    ))]
    fn load_from_file(
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<Self> {
        let columns = io::table::TableColumns::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        let inner = HighPerformancePointCloud::load_from_file(path, Some(columns))
            .map_err(|e| PyErr::from(e))?;
        Ok(PyPointCloud { inner })
    }

    /// 根据扩展名自动保存
    #[pyo3(signature = (
        path,
        x = None,
        y = None,
        z = None,
        intensity = None,
        rgb_r = None,
        rgb_g = None,
        rgb_b = None
    ))]
    fn save_to_file(
        &self,
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<()> {
        let columns = io::table::TableColumns::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        self.inner.save_to_file(path, Some(columns))
            .map_err(|e| PyErr::from(e))?;
        Ok(())
    }

    /// 获取内存占用（字节）
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// 转换为Python字典（包含numpy数组）
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        self.inner.to_numpy(py)
            .map_err(|e| PyErr::from(e))
    }

    /// 创建点云副本
    fn clone(&self) -> Self {
        PyPointCloud {
            inner: self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PointCloud(points={}, intensity={}, rgb={}, attributes={})",
            self.inner.point_count(),
            if self.inner.has_intensity() { "Yes" } else { "No" },
            if self.inner.has_rgb() { "Yes" } else { "No" },
            self.inner.attribute_names().len()
        )
    }
}

/// Python下采样策略枚举
#[pyclass(name = "DownsampleStrategy")]
pub struct PyDownsampleStrategy;

#[pymethods]
impl PyDownsampleStrategy {
    /// 随机采样策略
    #[classattr]
    fn RANDOM() -> i32 {
        0
    }

    /// 重心采样策略（最接近体素中心）
    #[classattr]
    fn CENTROID() -> i32 {
        1
    }
}
