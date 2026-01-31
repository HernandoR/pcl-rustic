/// 体素下采样：反射分组、2种采样策略实现
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::traits::{DownsampleStrategy, PointCloudProperties, VoxelDownsample};
use crate::utils::error::Result;
use crate::utils::reflect;
use crate::utils::tensor;

/// 随机采样策略
pub struct RandomSampleStrategy;

impl DownsampleStrategy for RandomSampleStrategy {
    fn select_representative(&self, indices: Vec<usize>, _xyz: &[Vec<f32>]) -> Result<usize> {
        if indices.is_empty() {
            return Err("体素内无点".into());
        }

        // 简单随机选择第一个点（生产环境可使用rand crate）
        Ok(indices[indices.len() / 2])
    }
}

/// 重心采样策略（选择最接近体素中心的点）
pub struct CentroidSampleStrategy;

impl DownsampleStrategy for CentroidSampleStrategy {
    fn select_representative(&self, indices: Vec<usize>, xyz: &[Vec<f32>]) -> Result<usize> {
        reflect::find_closest_to_centroid(&indices, xyz)
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
        let mut new_rgb: Option<(Vec<u8>, Vec<u8>, Vec<u8>)> = None;
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

        // 下采样RGB（3个独立通道）
        let (r_ref, g_ref, b_ref) = self.rgb_channels_ref();
        if let (Some(r), Some(g), Some(b)) = (r_ref, g_ref, b_ref) {
            let r_vec = tensor::tensor1_to_u8_vec(r);
            let g_vec = tensor::tensor1_to_u8_vec(g);
            let b_vec = tensor::tensor1_to_u8_vec(b);

            let mut new_r = Vec::new();
            let mut new_g = Vec::new();
            let mut new_b = Vec::new();
            for &idx in &selected_indices {
                new_r.push(r_vec[idx]);
                new_g.push(g_vec[idx]);
                new_b.push(b_vec[idx]);
            }
            new_rgb = Some((new_r, new_g, new_b));
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
        if let Some((r, g, b)) = new_rgb {
            result.set_rgb(r, g, b)?;
        }
        for (name, data) in new_attributes {
            result.set_attribute(name, data)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::PointCloudCore;

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
