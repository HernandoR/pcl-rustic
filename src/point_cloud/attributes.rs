/// 属性管理：intensity/RGB批量操作、自定义HashMap属性增/改
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::Result;
use crate::utils::tensor;

impl HighPerformancePointCloud {
    /// 清除所有自定义属性
    pub fn clear_attributes(&mut self) {
        self.attributes_mut().clear();
    }

    /// 批量替换自定义属性
    pub fn set_all_attributes(
        &mut self,
        attributes: std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<()> {
        // 验证所有属性的长度
        for (name, data) in &attributes {
            if data.len() != self.point_count() {
                return Err(format!(
                    "属性'{}'的长度不匹配: 期望{}，实际{}",
                    name,
                    self.point_count(),
                    data.len()
                ).into());
            }
        }

        // 替换所有属性
        let converted = attributes
            .into_iter()
            .map(|(name, data)| (name, tensor::intensity_to_tensor(data)))
            .collect();
        *self.attributes_mut() = converted;
        Ok(())
    }

    /// 获取intensity值（转向量）
    pub fn get_intensity_vec(&self) -> Option<Vec<f32>> {
        self.intensity_ref().map(tensor::tensor1_to_vec)
    }

    /// 获取RGB值（转向量）
    pub fn get_rgb_vec(&self) -> Option<Vec<Vec<u8>>> {
        self.rgb_ref().map(|r| r.clone())
    }

    /// 移除intensity
    pub fn remove_intensity(&mut self) {
        *self.intensity_mut() = None;
    }

    /// 移除RGB
    pub fn remove_rgb(&mut self) {
        *self.rgb_mut() = None;
    }

    /// 检查是否包含所有必要的属性
    pub fn has_attributes(&self, names: &[&str]) -> bool {
        let attrs = self.attributes_ref();
        names.iter().all(|name| attrs.contains_key(*name))
    }

    /// 获取所有属性的名称和元数据
    pub fn attribute_info(&self) -> Vec<(String, usize)> {
        self.attributes_ref()
            .iter()
            .map(|(name, data)| (name.clone(), tensor::tensor1_len(data)))
            .collect()
    }
}
