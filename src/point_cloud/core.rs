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

    // 可选：RGB颜色（3个独立通道，各为形状[M,]）
    rgb_r: Option<Tensor1>,
    rgb_g: Option<Tensor1>,
    rgb_b: Option<Tensor1>,

    // 自定义属性字典
    attributes: HashMap<String, Tensor1>,
}

impl HighPerformancePointCloud {
    /// 创建空的点云实例
    pub fn new() -> Self {
        Self {
            xyz: tensor::empty_xyz(),
            intensity: None,
            rgb_r: None,
            rgb_g: None,
            rgb_b: None,
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
            rgb_r: None,
            rgb_g: None,
            rgb_b: None,
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
            rgb_r: None,
            rgb_g: None,
            rgb_b: None,
            attributes: HashMap::new(),
        })
    }

    /// 从XYZ和RGB初始化（3个独立通道）
    pub fn from_xyz_rgb(xyz: Vec<Vec<f32>>, r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> Result<Self> {
        tensor::validate_xyz_shape(&xyz)?;
        let point_count = xyz.len();

        tensor::validate_rgb_channel_shape(&r, point_count)?;
        tensor::validate_rgb_channel_shape(&g, point_count)?;
        tensor::validate_rgb_channel_shape(&b, point_count)?;

        let xyz = tensor::xyz_to_tensor(xyz)?;

        Ok(Self {
            xyz,
            intensity: None,
            rgb_r: Some(tensor::rgb_channel_to_tensor(r)),
            rgb_g: Some(tensor::rgb_channel_to_tensor(g)),
            rgb_b: Some(tensor::rgb_channel_to_tensor(b)),
            attributes: HashMap::new(),
        })
    }

    /// 从XYZ、intensity和RGB初始化（3个独立通道）
    pub fn from_xyz_intensity_rgb(
        xyz: Vec<Vec<f32>>,
        intensity: Vec<f32>,
        r: Vec<u8>,
        g: Vec<u8>,
        b: Vec<u8>,
    ) -> Result<Self> {
        tensor::validate_xyz_shape(&xyz)?;
        let point_count = xyz.len();
        tensor::validate_intensity_shape(&intensity, point_count)?;

        tensor::validate_rgb_channel_shape(&r, point_count)?;
        tensor::validate_rgb_channel_shape(&g, point_count)?;
        tensor::validate_rgb_channel_shape(&b, point_count)?;

        let xyz = tensor::xyz_to_tensor(xyz)?;
        let intensity = tensor::intensity_to_tensor(intensity);

        Ok(Self {
            xyz,
            intensity: Some(intensity),
            rgb_r: Some(tensor::rgb_channel_to_tensor(r)),
            rgb_g: Some(tensor::rgb_channel_to_tensor(g)),
            rgb_b: Some(tensor::rgb_channel_to_tensor(b)),
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

    /// 获取内部RGB通道的可变引用（仅内部使用）
    pub(crate) fn rgb_channels_mut(
        &mut self,
    ) -> (
        &mut Option<Tensor1>,
        &mut Option<Tensor1>,
        &mut Option<Tensor1>,
    ) {
        (&mut self.rgb_r, &mut self.rgb_g, &mut self.rgb_b)
    }

    /// 获取内部RGB通道的不可变引用（仅内部使用）
    pub(crate) fn rgb_channels_ref(
        &self,
    ) -> (Option<&Tensor1>, Option<&Tensor1>, Option<&Tensor1>) {
        (
            self.rgb_r.as_ref(),
            self.rgb_g.as_ref(),
            self.rgb_b.as_ref(),
        )
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

        // RGB通道存储为f32
        if let Some(r) = &self.rgb_r {
            total += tensor::tensor1_len(r) * std::mem::size_of::<f32>();
        }
        if let Some(g) = &self.rgb_g {
            total += tensor::tensor1_len(g) * std::mem::size_of::<f32>();
        }
        if let Some(b) = &self.rgb_b {
            total += tensor::tensor1_len(b) * std::mem::size_of::<f32>();
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
        self.rgb_r.is_some() && self.rgb_g.is_some() && self.rgb_b.is_some()
    }

    fn get_intensity(&self) -> Option<Vec<f32>> {
        self.intensity.as_ref().map(tensor::tensor1_to_vec)
    }

    fn get_rgb(&self) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        match (&self.rgb_r, &self.rgb_g, &self.rgb_b) {
            (Some(r), Some(g), Some(b)) => Some((
                tensor::tensor1_to_u8_vec(r),
                tensor::tensor1_to_u8_vec(g),
                tensor::tensor1_to_u8_vec(b),
            )),
            _ => None,
        }
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

    fn set_rgb(&mut self, r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> Result<()> {
        let point_count = self.point_count();
        tensor::validate_rgb_channel_shape(&r, point_count)?;
        tensor::validate_rgb_channel_shape(&g, point_count)?;
        tensor::validate_rgb_channel_shape(&b, point_count)?;

        self.rgb_r = Some(tensor::rgb_channel_to_tensor(r));
        self.rgb_g = Some(tensor::rgb_channel_to_tensor(g));
        self.rgb_b = Some(tensor::rgb_channel_to_tensor(b));
        Ok(())
    }

    fn add_attribute(&mut self, name: String, data: Vec<f32>) -> Result<()> {
        if self.attributes.contains_key(&name) {
            return Err(format!("属性'{}'已存在", name).into());
        }
        tensor::validate_attribute_shape(&data, self.point_count())?;
        self.attributes
            .insert(name, tensor::intensity_to_tensor(data));
        Ok(())
    }

    fn set_attribute(&mut self, name: String, data: Vec<f32>) -> Result<()> {
        tensor::validate_attribute_shape(&data, self.point_count())?;
        self.attributes
            .insert(name, tensor::intensity_to_tensor(data));
        Ok(())
    }

    fn remove_attribute(&mut self, name: &str) -> Result<()> {
        if self.attributes.remove(name).is_none() {
            return Err(format!("属性'{}'不存在", name).into());
        }
        Ok(())
    }
}
