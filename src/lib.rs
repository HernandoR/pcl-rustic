mod interop;
mod io;
mod point_cloud;
mod traits;
mod utils;

use point_cloud::core::HighPerformancePointCloud;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use traits::{
    CoordinateTransform, DownsampleStrategy, PointCloudCore, PointCloudProperties, VoxelDownsample,
};

/// Python模块入口
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    /// 从 numpy XYZ 数组创建点云（支持 float32, float64, int32, int64）
    /// xyz: 形状为 [N, 3] 的 2D numpy 数组
    #[staticmethod]
    fn from_xyz(xyz: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    /// 从 numpy XYZ 和 intensity 数组创建点云
    /// xyz: 形状为 [N, 3] 的 2D numpy 数组
    /// intensity: 形状为 [N] 的 1D numpy 数组
    #[staticmethod]
    fn from_xyz_intensity(
        xyz: &Bound<'_, pyo3::PyAny>,
        intensity: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Self> {
        let mut inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        inner
            .set_intensity_from_array(intensity)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    /// 从 numpy XYZ 和 RGB 数组创建点云
    /// xyz: 形状为 [N, 3] 的 2D numpy 数组
    /// r, g, b: 形状为 [N] 的 1D numpy 数组
    #[staticmethod]
    fn from_xyz_rgb(
        xyz: &Bound<'_, pyo3::PyAny>,
        r: &Bound<'_, pyo3::PyAny>,
        g: &Bound<'_, pyo3::PyAny>,
        b: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Self> {
        let mut inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        inner.set_rgb_from_arrays(r, g, b).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    /// 从 numpy XYZ、intensity 和 RGB 数组创建点云
    #[staticmethod]
    fn from_xyz_intensity_rgb(
        xyz: &Bound<'_, pyo3::PyAny>,
        intensity: &Bound<'_, pyo3::PyAny>,
        r: &Bound<'_, pyo3::PyAny>,
        g: &Bound<'_, pyo3::PyAny>,
        b: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Self> {
        let mut inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        inner
            .set_intensity_from_array(intensity)
            .map_err(PyErr::from)?;
        inner.set_rgb_from_arrays(r, g, b).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    /// 从numpy字典创建点云
    #[staticmethod]
    fn from_dict(py: Python, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_numpy(py, data).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    /// 获取点数
    fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    /// 获取XYZ坐标（返回 numpy 数组字典中的 xyz）
    fn get_xyz(&self, py: Python) -> PyResult<PyObject> {
        use crate::utils::tensor;
        use numpy::ndarray::Array2;
        use numpy::IntoPyArray;

        let xyz_vec = tensor::tensor2_to_vec(self.inner.xyz_ref());
        let n = xyz_vec.len();
        let mut xyz_flat: Vec<f32> = Vec::with_capacity(n * 3);
        for row in &xyz_vec {
            xyz_flat.extend_from_slice(row);
        }
        let xyz_nd = Array2::from_shape_vec((n, 3), xyz_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("形状错误: {}", e)))?;
        let xyz_np = IntoPyArray::into_pyarray_bound(xyz_nd, py);
        Ok(xyz_np.into())
    }

    /// 检查是否有intensity
    fn has_intensity(&self) -> bool {
        self.inner.has_intensity()
    }

    /// 检查是否有RGB
    fn has_rgb(&self) -> bool {
        self.inner.has_rgb()
    }

    /// 获取 intensity（返回 numpy 数组）
    fn get_intensity(&self, py: Python) -> PyResult<Option<PyObject>> {
        use crate::utils::tensor;
        use numpy::ndarray::Array1;
        use numpy::IntoPyArray;

        if let Some(intensity) = self.inner.intensity_ref() {
            let vec = tensor::tensor1_to_vec(intensity);
            let nd = Array1::from_vec(vec);
            let np = IntoPyArray::into_pyarray_bound(nd, py);
            Ok(Some(np.into()))
        } else {
            Ok(None)
        }
    }

    /// 获取 RGB（返回 3 个 numpy 数组的元组）
    fn get_rgb(&self, py: Python) -> PyResult<Option<(PyObject, PyObject, PyObject)>> {
        use crate::utils::tensor;
        use numpy::ndarray::Array1;
        use numpy::IntoPyArray;

        let (r_ref, g_ref, b_ref) = self.inner.rgb_channels_ref();
        if let (Some(r), Some(g), Some(b)) = (r_ref, g_ref, b_ref) {
            let r_vec = tensor::tensor1_to_u8_vec(r);
            let g_vec = tensor::tensor1_to_u8_vec(g);
            let b_vec = tensor::tensor1_to_u8_vec(b);

            let r_nd = Array1::from_vec(r_vec);
            let g_nd = Array1::from_vec(g_vec);
            let b_nd = Array1::from_vec(b_vec);

            let r_np = IntoPyArray::into_pyarray_bound(r_nd, py);
            let g_np = IntoPyArray::into_pyarray_bound(g_nd, py);
            let b_np = IntoPyArray::into_pyarray_bound(b_nd, py);

            Ok(Some((r_np.into(), g_np.into(), b_np.into())))
        } else {
            Ok(None)
        }
    }

    /// 设置 intensity（从 numpy 数组）
    fn set_intensity(&mut self, intensity: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner
            .set_intensity_from_array(intensity)
            .map_err(PyErr::from)?;
        Ok(())
    }

    /// 设置 RGB（从 3 个 numpy 数组）
    fn set_rgb(
        &mut self,
        r: &Bound<'_, pyo3::PyAny>,
        g: &Bound<'_, pyo3::PyAny>,
        b: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner
            .set_rgb_from_arrays(r, g, b)
            .map_err(PyErr::from)?;
        Ok(())
    }

    /// 添加自定义属性（从 numpy 数组）
    fn add_attribute(&mut self, name: String, data: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        let tensor = read_attribute_array(data)?;
        let point_count = self.inner.point_count();
        let data_len = utils::tensor::tensor1_len(&tensor);
        if data_len != point_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "属性'{}'长度{}与点数{}不匹配",
                name, data_len, point_count
            )));
        }
        if self.inner.attributes_ref().contains_key(&name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "属性'{}'已存在",
                name
            )));
        }
        self.inner.attributes_mut().insert(name, tensor);
        Ok(())
    }

    /// 设置自定义属性（从 numpy 数组）
    fn set_attribute(&mut self, name: String, data: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        let tensor = read_attribute_array(data)?;
        let point_count = self.inner.point_count();
        let data_len = utils::tensor::tensor1_len(&tensor);
        if data_len != point_count {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "属性'{}'长度{}与点数{}不匹配",
                name, data_len, point_count
            )));
        }
        self.inner.attributes_mut().insert(name, tensor);
        Ok(())
    }

    /// 获取属性名列表
    fn attribute_names(&self) -> Vec<String> {
        self.inner.attribute_names()
    }

    /// 获取属性（返回 numpy 数组）
    fn get_attribute(&self, py: Python, name: &str) -> PyResult<Option<PyObject>> {
        use numpy::ndarray::Array1;
        use numpy::IntoPyArray;

        if let Some(attr) = self.inner.attributes_ref().get(name) {
            let vec = utils::tensor::tensor1_to_vec(attr);
            let nd = Array1::from_vec(vec);
            let np = IntoPyArray::into_pyarray_bound(nd, py);
            Ok(Some(np.into()))
        } else {
            Ok(None)
        }
    }

    /// 删除属性
    fn remove_attribute(&mut self, name: &str) -> PyResult<()> {
        self.inner.remove_attribute(name).map_err(PyErr::from)?;
        Ok(())
    }

    /// 清除所有自定义属性
    fn clear_attributes(&mut self) {
        self.inner.clear_attributes();
    }

    /// 批量设置所有自定义属性
    fn set_all_attributes(
        &mut self,
        attributes: std::collections::HashMap<String, Vec<f32>>,
    ) -> PyResult<()> {
        self.inner
            .set_all_attributes(attributes)
            .map_err(PyErr::from)?;
        Ok(())
    }

    /// 检查是否包含所有指定的属性
    fn has_attributes(&self, names: Vec<String>) -> bool {
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        self.inner.has_attributes(&name_refs)
    }

    /// 获取所有属性的名称和长度
    fn attribute_info(&self) -> Vec<(String, usize)> {
        self.inner.attribute_info()
    }

    /// 移除intensity
    fn remove_intensity(&mut self) {
        self.inner.remove_intensity();
    }

    /// 移除RGB
    fn remove_rgb(&mut self) {
        self.inner.remove_rgb();
    }

    /// 删除文件
    #[staticmethod]
    fn delete_file(path: &str) -> PyResult<()> {
        HighPerformancePointCloud::delete_file(path).map_err(PyErr::from)?;
        Ok(())
    }

    /// 坐标变换（矩阵）
    fn transform(&self, matrix: Vec<Vec<f32>>) -> PyResult<Self> {
        let result = self.inner.transform(matrix).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    /// 刚体变换（旋转+平移）
    fn rigid_transform(&self, rotation: Vec<Vec<f32>>, translation: Vec<f32>) -> PyResult<Self> {
        let result = self
            .inner
            .rigid_transform(rotation, translation)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    /// 体素下采样
    fn voxel_downsample(&self, voxel_size: f32, strategy: i32) -> PyResult<Self> {
        let strategy_impl: Box<dyn DownsampleStrategy> = match strategy {
            0 => Box::new(point_cloud::voxel::RandomSampleStrategy),
            1 => Box::new(point_cloud::voxel::CentroidSampleStrategy),
            _ => return Err(pyo3::exceptions::PyValueError::new_err("未知的采样策略")),
        };

        let result = self
            .inner
            .voxel_downsample(voxel_size, strategy_impl)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    /// 从LAS/LAZ文件读取
    #[staticmethod]
    fn from_las(path: &str) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_las_laz(path).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    /// 保存为LAS文件
    fn to_las(&self, path: &str, compress: bool) -> PyResult<()> {
        self.inner.to_las(path, compress).map_err(PyErr::from)?;
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
            .map_err(PyErr::from)?;
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
        let inner =
            HighPerformancePointCloud::from_table_parquet(path, columns).map_err(PyErr::from)?;
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
        self.inner
            .to_table_csv(path, delimiter, columns)
            .map_err(PyErr::from)?;
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
        self.inner
            .to_table_parquet(path, columns)
            .map_err(PyErr::from)?;
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
        let inner =
            HighPerformancePointCloud::load_from_file(path, Some(columns)).map_err(PyErr::from)?;
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
        self.inner
            .save_to_file(path, Some(columns))
            .map_err(PyErr::from)?;
        Ok(())
    }

    /// 获取内存占用（字节）
    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    /// 转换为Python字典（包含numpy数组）
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        self.inner.to_numpy(py).map_err(PyErr::from)
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
            if self.inner.has_intensity() {
                "Yes"
            } else {
                "No"
            },
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
    #[allow(non_snake_case)]
    fn RANDOM() -> i32 {
        0
    }

    /// 重心采样策略（最接近体素中心）
    #[classattr]
    #[allow(non_snake_case)]
    fn CENTROID() -> i32 {
        1
    }
}

// ============ 辅助函数：从 PyAny 读取 numpy 数组 ============

use numpy::{PyArray1, PyArrayMethods};
use utils::tensor::Tensor1;

/// 从 PyAny 读取 1D 数组作为属性，仅支持 f32 dtype
fn read_attribute_array(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Tensor1> {
    let arr = obj.downcast::<PyArray1<f32>>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "必须是dtype=float32的1D numpy数组，请使用 arr.astype(np.float32) 转换",
        )
    })?;

    let readonly = arr.readonly();
    let slice = readonly
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("无法读取数据，数组可能不连续"))?;
    Ok(utils::tensor::tensor1_from_slice(slice))
}
