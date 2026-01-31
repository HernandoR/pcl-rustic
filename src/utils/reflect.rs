use crate::utils::error::Result;
/// 反射工具：实现下采样的批量grouping逻辑
use std::collections::HashMap;

/// 通过反射和量化实现体素分组
/// 将点云按照给定的voxel_size进行分组
pub fn group_points_by_voxel(
    xyz: &[Vec<f32>],
    voxel_size: f32,
) -> Result<HashMap<String, Vec<usize>>> {
    if xyz.is_empty() {
        return Err("点云数据为空".into());
    }

    if voxel_size <= 0.0 {
        return Err("voxel_size必须大于0".into());
    }

    let mut voxel_groups: HashMap<String, Vec<usize>> = HashMap::new();

    for (idx, point) in xyz.iter().enumerate() {
        if point.len() != 3 {
            return Err(format!("点{}的维度不是3", idx).into());
        }

        // 计算点所属的体素坐标
        let voxel_x = (point[0] / voxel_size).floor() as i32;
        let voxel_y = (point[1] / voxel_size).floor() as i32;
        let voxel_z = (point[2] / voxel_size).floor() as i32;

        // 生成体素键
        let voxel_key = format!("{}_{}_{}", voxel_x, voxel_y, voxel_z);

        // 将点索引添加到对应体素组
        voxel_groups.entry(voxel_key).or_default().push(idx);
    }

    Ok(voxel_groups)
}

/// 获取体素内点的平均坐标（用于CENTROID策略）
pub fn compute_voxel_centroid(indices: &[usize], xyz: &[Vec<f32>]) -> Result<Vec<f32>> {
    if indices.is_empty() {
        return Err("体素内无点".into());
    }

    let mut centroid = vec![0.0; 3];

    for &idx in indices {
        if idx >= xyz.len() {
            return Err(format!("点索引{}超出范围", idx).into());
        }
        for j in 0..3 {
            centroid[j] += xyz[idx][j];
        }
    }

    for j in 0..3 {
        centroid[j] /= indices.len() as f32;
    }

    Ok(centroid)
}

/// 找到最接近体素中心的点
pub fn find_closest_to_centroid(indices: &[usize], xyz: &[Vec<f32>]) -> Result<usize> {
    if indices.is_empty() {
        return Err("体素内无点".into());
    }

    let centroid = compute_voxel_centroid(indices, xyz)?;
    let mut closest_idx = indices[0];
    let mut min_dist = f32::MAX;

    for &idx in indices {
        let point = &xyz[idx];
        let dist = (0..3)
            .map(|i| (point[i] - centroid[i]).powi(2))
            .sum::<f32>()
            .sqrt();

        if dist < min_dist {
            min_dist = dist;
            closest_idx = idx;
        }
    }

    Ok(closest_idx)
}
