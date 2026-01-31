/// NumPy互通：点云与numpy数组批量互转，零/低拷贝优化
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::PointCloudProperties;
use crate::utils::error::Result;
use crate::utils::tensor;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use numpy::ndarray::{Array1, Array2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

impl HighPerformancePointCloud {
    /// 转换为numpy数组字典
    /// 返回包含'xyz'、可选的'intensity'和'rgb'的字典
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Result<PyObject> {
        let dict = PyDict::new_bound(py);

        // 转换XYZ
        let xyz_array = tensor::tensor2_to_vec(self.xyz_ref());
        let n = xyz_array.len();
        let mut xyz_flat: Vec<f32> = Vec::with_capacity(n * 3);
        for row in &xyz_array {
            xyz_flat.extend_from_slice(row);
        }

        let xyz_nd = Array2::from_shape_vec((n, 3), xyz_flat)
            .map_err(|e| format!("XYZ shape error: {}", e))?;
        let xyz_np = IntoPyArray::into_pyarray_bound(xyz_nd, py);
        dict.set_item("xyz", xyz_np).map_err(|e| e.to_string())?;

        // 转换intensity（如果存在）
        if let Some(intensity) = self.intensity_ref() {
            let intensity_vec = tensor::tensor1_to_vec(intensity);
            let intensity_nd = Array1::from_vec(intensity_vec);
            let intensity_np = IntoPyArray::into_pyarray_bound(intensity_nd, py);
            dict.set_item("intensity", intensity_np).map_err(|e| e.to_string())?;
        }

        // 转换RGB（如果存在）
        if let Some(rgb) = self.rgb_ref() {
            let n = rgb.len();
            let mut rgb_flat: Vec<u8> = Vec::with_capacity(n * 3);
            for row in rgb {
                rgb_flat.extend_from_slice(row);
            }

            let rgb_nd = Array2::from_shape_vec((n, 3), rgb_flat)
                .map_err(|e| format!("RGB shape error: {}", e))?;
            let rgb_np = IntoPyArray::into_pyarray_bound(rgb_nd, py);
            dict.set_item("rgb", rgb_np).map_err(|e| e.to_string())?;
        }

        // 转换自定义属性
        for (name, attr) in self.attributes_ref() {
            let attr_vec = tensor::tensor1_to_vec(attr);
            let attr_nd = Array1::from_vec(attr_vec);
            let attr_np = IntoPyArray::into_pyarray_bound(attr_nd, py);
            dict.set_item(name.as_str(), attr_np).map_err(|e| e.to_string())?;
        }

        Ok(dict.into())
    }

    /// 从numpy数据创建点云
    /// data: 包含'xyz'等数据的字典
    pub fn from_numpy(_py: Python, data: &Bound<'_, PyDict>) -> Result<Self> {
        // 必须有xyz
        let xyz_obj = data.get_item("xyz")
            .map_err(|_| "获取xyz失败".to_string())?
            .ok_or("xyz字段缺失".to_string())?;

        let xyz_array: &Bound<'_, PyArray2<f32>> = xyz_obj
            .downcast()
            .map_err(|_| "xyz必须是float32的2D数组".to_string())?;

        let shape = xyz_array.shape();
        if shape[1] != 3 {
            return Err("xyz必须是[N,3]的形状".into());
        }

        let readonly = xyz_array.readonly();
        let xyz: Vec<Vec<f32>> = readonly
            .as_slice()
            .map_err(|_| "无法读取xyz数据".to_string())?
            .chunks(3)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut result = Self::from_xyz(xyz)?;

        // 可选：intensity
        if let Ok(Some(intensity_obj)) = data.get_item("intensity") {
            if let Ok(intensity_array) = intensity_obj.downcast::<PyArray1<f32>>() {
                let readonly = intensity_array.readonly();
                let intensity = readonly
                    .as_slice()
                    .map_err(|_| "无法读取intensity数据".to_string())?
                    .to_vec();
                result.set_intensity(intensity)?;
            }
        }

        // 可选：rgb
        if let Ok(Some(rgb_obj)) = data.get_item("rgb") {
            if let Ok(rgb_array) = rgb_obj.downcast::<PyArray2<u8>>() {
                let readonly = rgb_array.readonly();
                let rgb: Vec<Vec<u8>> = readonly
                    .as_slice()
                    .map_err(|_| "无法读取rgb数据".to_string())?
                    .chunks(3)
                    .map(|chunk| chunk.to_vec())
                    .collect();
                result.set_rgb(rgb)?;
            }
        }

        Ok(result)
    }
}
