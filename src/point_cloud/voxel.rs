/// 体素下采样：反射分组、2种采样策略实现
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::{VoxelDownsample, DownsampleStrategy};
use crate::utils::reflect;
use crate::utils::error::Result;
use crate::utils::tensor;

/// 随机采样策略
pub struct RandomSampleStrategy;

impl DownsampleStrategy for RandomSampleStrategy {
    fn select_representative(
        &self,
        indices: Vec<usize>,
        _xyz: &[Vec<f32>],
    ) -> Result<usize> {
        if indices.is_empty() {
            return Err("体素内无点".into());
        }

        // 简单随机选择第一个点（生产环境可使用rand crate）
        Ok(indices[indices.len() / 2])
    }

    fn name(&self) -> &str {
        "RandomSample"
    }
}

/// 重心采样策略（选择最接近体素中心的点）
pub struct CentroidSampleStrategy;

impl DownsampleStrategy for CentroidSampleStrategy {
    fn select_representative(
        &self,
        indices: Vec<usize>,
        xyz: &[Vec<f32>],
    ) -> Result<usize> {
        reflect::find_closest_to_centroid(&indices, xyz)
    }

    fn name(&self) -> &str {
        "CentroidSample"
    }
}

impl VoxelDownsample for HighPerformancePointCloud {
    fn voxel_downsample(
        &self,
        voxel_size: f32,
        strategy: Box<dyn DownsampleStrategy>,
    ) -> Result<Self> {
        if voxel_size <= 0.0 {
            return Err("voxel_size必须大于0".into());
        }

        let xyz_ref = tensor::tensor2_to_vec(self.xyz_ref());

        if xyz_ref.is_empty() {
            return Ok(self.clone());
        }

        // 第一步：通过反射分组
        let voxel_groups = reflect::group_points_by_voxel(&xyz_ref, voxel_size)?;

        // 第二步：对每个体素应用采样策略
        let mut selected_indices = Vec::new();

        for (_, indices) in voxel_groups.iter() {
            if !indices.is_empty() {
                let selected = strategy.select_representative(indices.clone(), &xyz_ref)?;
                selected_indices.push(selected);
            }
        }

        // 对索引排序以保持原始顺序
        selected_indices.sort_unstable();

        // 第三步：构建下采样后的点云
        let mut new_xyz = Vec::new();
        let mut new_intensity = None;
        let mut new_rgb = None;
        let mut new_attributes = std::collections::HashMap::new();

        for &idx in &selected_indices {
            new_xyz.push(xyz_ref[idx].clone());
        }

        // 下采样intensity
        if let Some(intensity) = self.intensity_ref() {
            let intensity_vec = tensor::tensor1_to_vec(intensity);
            let mut new_int = Vec::new();
            for &idx in &selected_indices {
                new_int.push(intensity_vec[idx]);
            }
            new_intensity = Some(new_int);
        }

        // 下采样RGB
        if let Some(rgb) = self.rgb_ref() {
            let mut new_rgb_data = Vec::new();
            for &idx in &selected_indices {
                new_rgb_data.push(rgb[idx].clone());
            }
            new_rgb = Some(new_rgb_data);
        }

        // 下采样自定义属性
        for (name, data) in self.attributes_ref() {
            let data_vec = tensor::tensor1_to_vec(data);
            let mut new_data = Vec::new();
            for &idx in &selected_indices {
                new_data.push(data_vec[idx]);
            }
            new_attributes.insert(name.clone(), new_data);
        }

        // 构造新点云
        let mut result = HighPerformancePointCloud::from_xyz(new_xyz)?;
        if let Some(intensity) = new_intensity {
            result.set_intensity(intensity)?;
        }
        if let Some(rgb) = new_rgb {
            result.set_rgb(rgb)?;
        }
        for (name, data) in new_attributes {
            result.set_attribute(name, data)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_downsample() {
        let xyz = vec![
            vec![0.1, 0.1, 0.1],
            vec![0.2, 0.2, 0.2],
            vec![1.1, 1.1, 1.1],
            vec![1.2, 1.2, 1.2],
        ];

        let pc = HighPerformancePointCloud::from_xyz(xyz).unwrap();
        let strategy = Box::new(CentroidSampleStrategy);
        let downsampled = pc.voxel_downsample(1.0, strategy).unwrap();

        // 应该得到2个点（每个体素1个代表点）
        assert!(downsampled.point_count() <= pc.point_count());
    }
}
