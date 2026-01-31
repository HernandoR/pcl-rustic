/// NumPy互通：点云与numpy数组批量互转，零/低拷贝优化
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::PointCloudCore;
use crate::utils::error::Result;
use crate::utils::tensor;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

impl HighPerformancePointCloud {
    /// 转换为numpy数组字典
    /// 返回包含'xyz'、可选的'intensity'和'rgb' (r, g, b分离)的字典
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Result<Py<PyAny>> {
        let dict = PyDict::new(py);

        // 转换XYZ
        let xyz_array = tensor::tensor2_to_vec(self.xyz_ref());
        let n = xyz_array.len();
        let mut xyz_flat: Vec<f32> = Vec::with_capacity(n * 3);
        for row in &xyz_array {
            xyz_flat.extend_from_slice(row);
        }

        let xyz_nd = Array2::from_shape_vec((n, 3), xyz_flat)
            .map_err(|e| format!("XYZ shape error: {}", e))?;
        let xyz_np = IntoPyArray::into_pyarray(xyz_nd, py);
        dict.set_item("xyz", xyz_np)
            .map_err(|e: PyErr| e.to_string())?;

        // 转换intensity（如果存在）
        if let Some(intensity) = self.intensity_ref() {
            let intensity_vec = tensor::tensor1_to_vec(intensity);
            let intensity_nd = Array1::from_vec(intensity_vec);
            let intensity_np = IntoPyArray::into_pyarray(intensity_nd, py);
            dict.set_item("intensity", intensity_np)
                .map_err(|e: PyErr| e.to_string())?;
        }

        // 转换RGB（如果存在）- 分离的R/G/B通道
        let (r_ref, g_ref, b_ref) = self.rgb_channels_ref();
        if let (Some(r), Some(g), Some(b)) = (r_ref, g_ref, b_ref) {
            let r_vec = tensor::tensor1_to_u8_vec(r);
            let g_vec = tensor::tensor1_to_u8_vec(g);
            let b_vec = tensor::tensor1_to_u8_vec(b);

            let r_nd = Array1::from_vec(r_vec);
            let g_nd = Array1::from_vec(g_vec);
            let b_nd = Array1::from_vec(b_vec);

            let r_np = IntoPyArray::into_pyarray(r_nd, py);
            let g_np = IntoPyArray::into_pyarray(g_nd, py);
            let b_np = IntoPyArray::into_pyarray(b_nd, py);

            dict.set_item("r", r_np).map_err(|e: PyErr| e.to_string())?;
            dict.set_item("g", g_np).map_err(|e: PyErr| e.to_string())?;
            dict.set_item("b", b_np).map_err(|e: PyErr| e.to_string())?;
        }

        // 转换自定义属性
        for (name, attr) in self.attributes_ref() {
            let attr_vec = tensor::tensor1_to_vec(attr);
            let attr_nd = Array1::from_vec(attr_vec);
            let attr_np = IntoPyArray::into_pyarray(attr_nd, py);
            dict.set_item(name.as_str(), attr_np)
                .map_err(|e: PyErr| e.to_string())?;
        }

        Ok(dict.into())
    }

    /// 从 PyAny（numpy array）读取 XYZ 数据，支持多种 dtype
    /// 返回创建的 HighPerformancePointCloud
    pub fn from_xyz_array(xyz_obj: &Bound<'_, pyo3::PyAny>) -> Result<Self> {
        let xyz = read_xyz_from_pyany(xyz_obj)?;
        Self::from_tensor_xyz(xyz)
    }

    /// 从numpy数据字典创建点云，支持多种 dtype
    pub fn from_numpy(_py: Python, data: &Bound<'_, PyDict>) -> Result<Self> {
        // 必须有xyz
        let xyz_obj = data
            .get_item("xyz")
            .map_err(|_| "获取xyz失败".to_string())?
            .ok_or("xyz字段缺失".to_string())?;

        let xyz = read_xyz_from_pyany(&xyz_obj)?;
        let mut result = Self::from_tensor_xyz(xyz)?;

        // 可选：intensity（支持多种 dtype）
        if let Ok(Some(intensity_obj)) = data.get_item("intensity") {
            let intensity = read_1d_array_from_pyany(&intensity_obj)?;
            if tensor::tensor1_len(&intensity) == result.point_count() {
                *result.intensity_mut() = Some(intensity);
            }
        }

        // 可选：rgb（分离的r、g、b通道，支持多种 dtype）
        let r_item = data.get_item("r");
        let g_item = data.get_item("g");
        let b_item = data.get_item("b");

        if let (Ok(Some(r_obj)), Ok(Some(g_obj)), Ok(Some(b_obj))) = (r_item, g_item, b_item) {
            let r = read_1d_array_from_pyany(&r_obj)?;
            let g = read_1d_array_from_pyany(&g_obj)?;
            let b = read_1d_array_from_pyany(&b_obj)?;

            let point_count = result.point_count();
            if tensor::tensor1_len(&r) == point_count
                && tensor::tensor1_len(&g) == point_count
                && tensor::tensor1_len(&b) == point_count
            {
                let (r_mut, g_mut, b_mut) = result.rgb_channels_mut();
                *r_mut = Some(r);
                *g_mut = Some(g);
                *b_mut = Some(b);
            }
        }

        Ok(result)
    }

    /// 设置 intensity（从 PyAny numpy array）
    pub fn set_intensity_from_array(&mut self, arr: &Bound<'_, pyo3::PyAny>) -> Result<()> {
        let intensity = read_1d_array_from_pyany(arr)?;
        if tensor::tensor1_len(&intensity) != self.point_count() {
            return Err(format!(
                "Intensity长度{}与点数{}不匹配",
                tensor::tensor1_len(&intensity),
                self.point_count()
            )
            .into());
        }
        *self.intensity_mut() = Some(intensity);
        Ok(())
    }

    /// 设置 RGB（从 PyAny numpy arrays）
    pub fn set_rgb_from_arrays(
        &mut self,
        r_arr: &Bound<'_, pyo3::PyAny>,
        g_arr: &Bound<'_, pyo3::PyAny>,
        b_arr: &Bound<'_, pyo3::PyAny>,
    ) -> Result<()> {
        let r = read_1d_array_from_pyany(r_arr)?;
        let g = read_1d_array_from_pyany(g_arr)?;
        let b = read_1d_array_from_pyany(b_arr)?;

        let point_count = self.point_count();
        let r_len = tensor::tensor1_len(&r);
        let g_len = tensor::tensor1_len(&g);
        let b_len = tensor::tensor1_len(&b);

        if r_len != point_count || g_len != point_count || b_len != point_count {
            return Err(format!(
                "RGB通道长度({},{},{})与点数{}不匹配",
                r_len, g_len, b_len, point_count
            )
            .into());
        }

        let (r_mut, g_mut, b_mut) = self.rgb_channels_mut();
        *r_mut = Some(r);
        *g_mut = Some(g);
        *b_mut = Some(b);
        Ok(())
    }
}

// ============ 内部辅助函数：从 PyAny 读取数据 ============

use crate::utils::tensor::{Tensor1, Tensor2};
use numpy::{PyArray1, PyArray2};

/// 从 PyAny 读取 2D XYZ 数组，仅支持 f32 dtype
fn read_xyz_from_pyany(obj: &Bound<'_, pyo3::PyAny>) -> Result<Tensor2> {
    let arr = obj
        .cast::<PyArray2<f32>>()
        .map_err(|_| "xyz必须是dtype=float32的2D numpy数组，请使用 arr.astype(np.float32) 转换")?;

    let shape = arr.shape();
    if shape[1] != 3 {
        return Err(format!("XYZ必须是[N,3]的形状，实际为[{},{}]", shape[0], shape[1]).into());
    }

    let readonly = arr.readonly();
    let slice = readonly
        .as_slice()
        .map_err(|_| "无法读取xyz数据，数组可能不连续")?;
    // tensor::xyz_from_slice(slice)
    tensor::tensor2_from_slice(slice, shape[0], shape[1])
}

/// 从 PyAny 读取 1D 数组，仅支持 f32 dtype
fn read_1d_array_from_pyany(obj: &Bound<'_, pyo3::PyAny>) -> Result<Tensor1> {
    let arr = obj
        .cast::<PyArray1<f32>>()
        .map_err(|_| "必须是dtype=float32的1D numpy数组，请使用 arr.astype(np.float32) 转换")?;

    let readonly = arr.readonly();
    let slice = readonly
        .as_slice()
        .map_err(|_| "无法读取数据，数组可能不连续")?;
    Ok(tensor::tensor1_from_slice(slice))
}
