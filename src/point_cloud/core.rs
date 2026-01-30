use crate::traits::{PointCloudCore, PointCloudProperties};
use crate::utils::error::Result;
use crate::utils::tensor;
use crate::utils::tensor::{Tensor1, Tensor2};
/// 点云核心Struct定义、基础生命周期方法
use std::collections::HashMap;

/// 高性能点云结构体
/// 所有字段私有，仅通过Trait方法/公有接口暴露批量操作
#[derive(Clone)]
pub struct HighPerformancePointCloud {
    // 必选：XYZ三维坐标（形状[M,3]，M为点数）
    xyz: Tensor2,

    // 可选：强度值（形状[M,]）
    intensity: Option<Tensor1>,

    // 可选：RGB颜色（形状[M,3]）
    rgb: Option<Vec<Vec<u8>>>,

    // 自定义属性字典
    attributes: HashMap<String, Tensor1>,
}

impl HighPerformancePointCloud {
    /// 创建空的点云实例
    pub fn new() -> Self {
        Self {
            xyz: tensor::empty_xyz(),
            intensity: None,
            rgb: None,
            attributes: HashMap::new(),
        }
    }

    /// 从XYZ张量初始化点云
    /// xyz: [M,3]形状的坐标矩阵
    pub fn from_xyz(xyz: Vec<Vec<f32>>) -> Result<Self> {
        tensor::validate_xyz_shape(&xyz)?;

        let xyz = tensor::xyz_to_tensor(xyz)?;

        Ok(Self {
            xyz,
            intensity: None,
            rgb: None,
            attributes: HashMap::new(),
        })
    }

    /// 从XYZ和intensity初始化
    pub fn from_xyz_intensity(xyz: Vec<Vec<f32>>, intensity: Vec<f32>) -> Result<Self> {
        tensor::validate_xyz_shape(&xyz)?;
        let point_count = xyz.len();
        tensor::validate_intensity_shape(&intensity, point_count)?;

        let xyz = tensor::xyz_to_tensor(xyz)?;
        let intensity = tensor::intensity_to_tensor(intensity);

        Ok(Self {
            xyz,
            intensity: Some(intensity),
            rgb: None,
            attributes: HashMap::new(),
        })
    }

    /// 从XYZ和RGB初始化
    pub fn from_xyz_rgb(xyz: Vec<Vec<f32>>, rgb: Vec<Vec<u8>>) -> Result<Self> {
        tensor::validate_xyz_shape(&xyz)?;
        let point_count = xyz.len();

        if rgb.len() != point_count {
            return Err(format!("RGB点数不匹配: 期望{}，实际{}", point_count, rgb.len()).into());
        }

        tensor::validate_rgb_shape(&rgb)?;

        let xyz = tensor::xyz_to_tensor(xyz)?;

        Ok(Self {
            xyz,
            intensity: None,
            rgb: Some(rgb),
            attributes: HashMap::new(),
        })
    }

    /// 从XYZ、intensity和RGB初始化
    pub fn from_xyz_intensity_rgb(
        xyz: Vec<Vec<f32>>,
        intensity: Vec<f32>,
        rgb: Vec<Vec<u8>>,
    ) -> Result<Self> {
        tensor::validate_xyz_shape(&xyz)?;
        let point_count = xyz.len();
        tensor::validate_intensity_shape(&intensity, point_count)?;

        if rgb.len() != point_count {
            return Err(format!("RGB点数不匹配: 期望{}，实际{}", point_count, rgb.len()).into());
        }

        tensor::validate_rgb_shape(&rgb)?;

        let xyz = tensor::xyz_to_tensor(xyz)?;
        let intensity = tensor::intensity_to_tensor(intensity);

        Ok(Self {
            xyz,
            intensity: Some(intensity),
            rgb: Some(rgb),
            attributes: HashMap::new(),
        })
    }

    /// 获取内部XYZ的可变引用（仅内部使用）
    pub(crate) fn xyz_mut(&mut self) -> &mut Tensor2 {
        &mut self.xyz
    }

    /// 获取内部XYZ的不可变引用（仅内部使用）
    pub(crate) fn xyz_ref(&self) -> &Tensor2 {
        &self.xyz
    }

    /// 获取内部intensity的可变引用（仅内部使用）
    pub(crate) fn intensity_mut(&mut self) -> &mut Option<Tensor1> {
        &mut self.intensity
    }

    /// 获取内部intensity的不可变引用（仅内部使用）
    pub(crate) fn intensity_ref(&self) -> Option<&Tensor1> {
        self.intensity.as_ref()
    }

    /// 获取内部RGB的可变引用（仅内部使用）
    pub(crate) fn rgb_mut(&mut self) -> &mut Option<Vec<Vec<u8>>> {
        &mut self.rgb
    }

    /// 获取内部RGB的不可变引用（仅内部使用）
    pub(crate) fn rgb_ref(&self) -> Option<&Vec<Vec<u8>>> {
        self.rgb.as_ref()
    }

    /// 获取内部属性字典的可变引用（仅内部使用）
    pub(crate) fn attributes_mut(&mut self) -> &mut HashMap<String, Tensor1> {
        &mut self.attributes
    }

    /// 获取内部属性字典的不可变引用（仅内部使用）
    pub(crate) fn attributes_ref(&self) -> &HashMap<String, Tensor1> {
        &self.attributes
    }

    /// 内存占用估算（字节）
    pub fn memory_usage(&self) -> usize {
        let mut total = self.point_count() * 3 * std::mem::size_of::<f32>();

        if let Some(intensity) = &self.intensity {
            total += tensor::tensor1_len(intensity) * std::mem::size_of::<f32>();
        }

        if let Some(rgb) = &self.rgb {
            total += rgb.len() * 3 * std::mem::size_of::<u8>();
        }

        for (_, data) in &self.attributes {
            total += tensor::tensor1_len(data) * std::mem::size_of::<f32>();
        }

        total
    }
}

impl Default for HighPerformancePointCloud {
    fn default() -> Self {
        Self::new()
    }
}

impl PointCloudCore for HighPerformancePointCloud {
    fn get_xyz(&self) -> Vec<Vec<f32>> {
        tensor::tensor2_to_vec(&self.xyz)
    }

    fn point_count(&self) -> usize {
        tensor::tensor2_rows(&self.xyz)
    }

    fn has_intensity(&self) -> bool {
        self.intensity.is_some()
    }

    fn has_rgb(&self) -> bool {
        self.rgb.is_some()
    }

    fn get_intensity(&self) -> Option<Vec<f32>> {
        self.intensity.as_ref().map(tensor::tensor1_to_vec)
    }

    fn get_rgb(&self) -> Option<Vec<Vec<u8>>> {
        self.rgb.clone()
    }

    fn attribute_names(&self) -> Vec<String> {
        self.attributes.keys().cloned().collect()
    }

    fn get_attribute(&self, name: &str) -> Option<Vec<f32>> {
        self.attributes.get(name).map(tensor::tensor1_to_vec)
    }
}

impl PointCloudProperties for HighPerformancePointCloud {
    fn set_intensity(&mut self, intensity: Vec<f32>) -> Result<()> {
        tensor::validate_intensity_shape(&intensity, self.point_count())?;
        self.intensity = Some(tensor::intensity_to_tensor(intensity));
        Ok(())
    }

    fn set_rgb(&mut self, rgb: Vec<Vec<u8>>) -> Result<()> {
        if rgb.len() != self.point_count() {
            return Err(format!("RGB点数不匹配: 期望{}，实际{}", self.point_count(), rgb.len()).into());
        }
        tensor::validate_rgb_shape(&rgb)?;
        self.rgb = Some(rgb);
        Ok(())
    }

    fn add_attribute(&mut self, name: String, data: Vec<f32>) -> Result<()> {
        if self.attributes.contains_key(&name) {
            return Err(format!("属性'{}'已存在", name).into());
        }
        tensor::validate_attribute_shape(&data, self.point_count())?;
        self.attributes.insert(name, tensor::intensity_to_tensor(data));
        Ok(())
    }

    fn set_attribute(&mut self, name: String, data: Vec<f32>) -> Result<()> {
        tensor::validate_attribute_shape(&data, self.point_count())?;
        self.attributes.insert(name, tensor::intensity_to_tensor(data));
        Ok(())
    }

    fn remove_attribute(&mut self, name: &str) -> Result<()> {
        if self.attributes.remove(name).is_none() {
            return Err(format!("属性'{}'不存在", name).into());
        }
        Ok(())
    }
}
