/// NumPy互通：点云与numpy数组批量互转，零/低拷贝优化
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::{PointCloudCore, PointCloudProperties};
use crate::utils::error::Result;
use crate::utils::tensor;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

impl HighPerformancePointCloud {
    /// 转换为numpy数组字典
    /// 返回包含'xyz'、可选的'intensity'和'rgb'的字典
    pub fn to_numpy(&self, py: Python) -> Result<PyObject> {
        let mut data = std::collections::HashMap::new();

        // 转换XYZ
        let xyz_array = tensor::tensor2_to_vec(self.xyz_ref());
        let xyz_flat: Vec<f32> = xyz_array.into_iter().flatten().collect();

        let xyz_np = PyArray2::<f32>::new(py, [self.point_count(), 3], false);

        unsafe {
            let mut slice = xyz_np.as_array_mut();
            for (i, &v) in xyz_flat.iter().enumerate() {
                slice.flat[i] = v;
            }
        }

        data.insert("xyz", xyz_np.to_object(py));

        // 转换intensity（如果存在）
        if let Some(intensity) = self.intensity_ref() {
            let intensity_vec = tensor::tensor1_to_vec(intensity);
            let intensity_np = PyArray1::<f32>::new(py, intensity_vec.len(), false);
            unsafe {
                let mut slice = intensity_np.as_array_mut();
                for (i, &v) in intensity_vec.iter().enumerate() {
                    slice[i] = v;
                }
            }
            data.insert("intensity", intensity_np.to_object(py));
        }

        // 转换RGB（如果存在）
        if let Some(rgb) = self.rgb_ref() {
            let rgb_flat: Vec<u8> = rgb.iter().flat_map(|row| row.iter().copied()).collect();

            let rgb_np = PyArray2::<u8>::new(py, [self.point_count(), 3], false);

            unsafe {
                let mut slice = rgb_np.as_array_mut();
                for (i, &v) in rgb_flat.iter().enumerate() {
                    slice.flat[i] = v;
                }
            }

            data.insert("rgb", rgb_np.to_object(py));
        }

        // 转换自定义属性
        for (name, attr) in self.attributes_ref() {
            let attr_vec = tensor::tensor1_to_vec(attr);
            let attr_np = PyArray1::<f32>::new(py, attr_vec.len(), false);
            unsafe {
                let mut slice = attr_np.as_array_mut();
                for (i, &v) in attr_vec.iter().enumerate() {
                    slice[i] = v;
                }
            }
            data.insert(name.clone(), attr_np.to_object(py));
        }

        Ok(PyDict::from_sequence(py, pyo3::PyDict::new(py).to_object(py))?.to_object(py))
    }

    /// 从numpy数据创建点云
    /// data: 包含'xyz'等数据的字典
    pub fn from_numpy(py: Python, data: PyObject) -> Result<Self> {
        let dict = pyo3::PyDict::extract(&data).map_err(|_| "输入必须是字典".to_string())?;

        // 必须有xyz
        let xyz_obj = dict.get_item("xyz").ok_or("xyz字段缺失".to_string())?;

        let xyz_array = xyz_obj
            .extract::<&PyArray2<f32>>()
            .map_err(|_| "xyz必须是float32的2D数组".to_string())?;

        let shape = xyz_array.dims();
        if shape[1] != 3 {
            return Err("xyz必须是[N,3]的形状".into());
        }

        let xyz = unsafe {
            xyz_array
                .as_slice()
                .map_err(|_| "无法读取xyz数据".to_string())?
                .chunks(3)
                .map(|chunk| chunk.to_vec())
                .collect::<Vec<_>>()
        };

        let mut result = Self::from_xyz(xyz)?;

        // 可选：intensity
        if let Some(intensity_obj) = dict.get_item("intensity") {
            if let Ok(intensity_array) = intensity_obj.extract::<&PyArray1<f32>>() {
                let intensity = unsafe {
                    intensity_array
                        .as_slice()
                        .map_err(|_| "无法读取intensity数据".to_string())?
                        .to_vec()
                };
                result.set_intensity(intensity)?;
            }
        }

        // 可选：rgb
        if let Some(rgb_obj) = dict.get_item("rgb") {
            if let Ok(rgb_array) = rgb_obj.extract::<&PyArray2<u8>>() {
                let rgb = unsafe {
                    rgb_array
                        .as_slice()
                        .map_err(|_| "无法读取rgb数据".to_string())?
                        .chunks(3)
                        .map(|chunk| chunk.to_vec())
                        .collect::<Vec<_>>()
                };
                result.set_rgb(rgb)?;
            }
        }

        Ok(result)
    }
}
