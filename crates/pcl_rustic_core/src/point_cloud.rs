use crate::point::Point;
use log_once::debug_once;
use nalgebra::{Matrix4, Vector4};
use polars::prelude::*;
use std::collections::HashMap;

/// A table-based point cloud using polars DataFrame
/// Each attribute is stored as a column
#[derive(Debug, Clone)]
pub struct TablePointCloud {
    data: DataFrame,
}

impl TablePointCloud {
    /// Create a new empty TablePointCloud
    pub fn new() -> Result<Self, PolarsError> {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let z: Vec<f64> = vec![];

        let df = DataFrame::new(vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y).into(),
            Series::new("z".into(), z).into(),
        ])?;

        Ok(TablePointCloud { data: df })
    }

    /// Create a TablePointCloud from vectors of coordinates
    pub fn from_xyz(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> Result<Self, PolarsError> {
        if x.len() != y.len() || y.len() != z.len() {
            return Err(PolarsError::ShapeMismatch(
                "x, y, z vectors must have the same length".into(),
            ));
        }

        let df = DataFrame::new(vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y).into(),
            Series::new("z".into(), z).into(),
        ])?;

        Ok(TablePointCloud { data: df })
    }

    /// Create a TablePointCloud from a vector of Points
    pub fn from_points(points: Vec<Point<f64>>) -> Result<Self, PolarsError> {
        if points.is_empty() {
            return Self::new();
        }

        let x: Vec<f64> = points.iter().map(|p| p.x).collect();
        let y: Vec<f64> = points.iter().map(|p| p.y).collect();
        let z: Vec<f64> = points.iter().map(|p| p.z).collect();

        let mut series = vec![
            Series::new("x".into(), x).into(),
            Series::new("y".into(), y).into(),
            Series::new("z".into(), z).into(),
        ];

        // Collect all unique attribute keys
        let mut all_keys: Vec<String> = points
            .iter()
            .flat_map(|p| p.attributes.keys().cloned())
            .collect();
        all_keys.sort();
        all_keys.dedup();

        // Add columns for each attribute
        for key in all_keys {
            let values: Vec<f64> = points
                .iter()
                .map(|p| p.attributes.get(&key).copied().unwrap_or(f64::NAN))
                .collect();
            series.push(Series::new(key.as_str().into(), values).into());
        }

        let df = DataFrame::new(series)?;
        Ok(TablePointCloud { data: df })
    }

    /// Transform the point cloud using a 4x4 homogeneous transformation matrix
    /// Performs right multiplication: P_b = T_a2b @ P_a
    /// where each point is represented in homogeneous coordinates [x, y, z, 1]
    pub fn transform(&self, transform: &Matrix4<f64>) -> Result<Self, PolarsError> {
        let x_values = self.x()?;
        let y_values = self.y()?;
        let z_values = self.z()?;

        let mut transformed_x = Vec::with_capacity(self.len());
        let mut transformed_y = Vec::with_capacity(self.len());
        let mut transformed_z = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            // Create homogeneous point [x, y, z, 1]
            let point_homo = Vector4::new(x_values[i], y_values[i], z_values[i], 1.0);

            // Apply transformation: P' = T * P
            let transformed_homo = transform * point_homo;

            // Convert back from homogeneous coordinates (divide by w)
            let w = transformed_homo[3];
            transformed_x.push(transformed_homo[0] / w);
            transformed_y.push(transformed_homo[1] / w);
            transformed_z.push(transformed_homo[2] / w);
        }

        // Create new point cloud with transformed coordinates
        let mut new_cloud = Self::from_xyz(transformed_x, transformed_y, transformed_z)?;

        // Copy over all attribute columns (they don't change with spatial transformation)
        for col_name in self.data.get_column_names() {
            if col_name != "x" && col_name != "y" && col_name != "z" {
                if let Ok(series) = self.data.column(col_name) {
                    new_cloud
                        .data
                        .with_column(series.clone())
                        .map_err(|e| PolarsError::ComputeError(format!("{}", e).into()))?;
                }
            }
        }

        Ok(new_cloud)
    }

    /// Get the number of points in the cloud
    pub fn len(&self) -> usize {
        self.data.height()
    }

    /// Check if the point cloud is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a column (attribute) to the point cloud
    pub fn add_attribute(&mut self, name: &str, values: Vec<f64>) -> Result<(), PolarsError> {
        if values.len() != self.len() {
            return Err(PolarsError::ShapeMismatch(
                format!(
                    "Attribute length {} does not match point cloud length {}",
                    values.len(),
                    self.len()
                )
                .into(),
            ));
        }
        // if attribute already exists, raise an error
        if self
            .data
            .get_column_names()
            .iter()
            .any(|s| s.as_str() == name)
        {
            return Err(PolarsError::Duplicate(
                format!("Attribute '{}' already exists", name).into(),
            ));
        }

        let series = Series::new(name.into(), values);
        self.data.with_column(series)?;
        Ok(())
    }

    /// Get a reference to the underlying DataFrame
    pub fn data(&self) -> &DataFrame {
        &self.data
    }

    /// Get x coordinates as a vector
    pub fn x(&self) -> Result<Vec<f64>, PolarsError> {
        let series = self.data.column("x")?;
        Ok(series
            .f64()?
            .to_vec()
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect())
    }

    /// Get y coordinates as a vector
    pub fn y(&self) -> Result<Vec<f64>, PolarsError> {
        let series = self.data.column("y")?;
        Ok(series
            .f64()?
            .to_vec()
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect())
    }

    /// Get z coordinates as a vector
    pub fn z(&self) -> Result<Vec<f64>, PolarsError> {
        let series = self.data.column("z")?;
        Ok(series
            .f64()?
            .to_vec()
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect())
    }

    /// Get a point at a specific index
    pub fn get_point(&self, index: usize) -> Result<Point<f64>, PolarsError> {
        if index >= self.len() {
            return Err(PolarsError::OutOfBounds(
                format!("Index {} out of bounds for length {}", index, self.len()).into(),
            ));
        }

        let x = self.data.column("x")?.f64()?.get(index).unwrap_or(0.0);
        let y = self.data.column("y")?.f64()?.get(index).unwrap_or(0.0);
        let z = self.data.column("z")?.f64()?.get(index).unwrap_or(0.0);

        let mut attributes = HashMap::new();
        for col_name in self.data.get_column_names() {
            // Skip coordinate columns
            if ["x", "y", "z"].contains(&col_name.as_str()) {
                debug_once!("Skipping coordinate column '{}'", col_name);
                continue;
            }
            // Get the column as series
            let series = match self.data.column(col_name) {
                Ok(series) => series,
                Err(_) => {
                    debug_once!("Column '{}' could not be accessed, skipping", col_name);
                    continue;
                }
            };

            // Convert to f64 series
            let f64_series = match series.f64() {
                Ok(series) => series,
                Err(_) => {
                    debug_once!(
                        "Column '{}' could not be converted to f64, skipping",
                        col_name
                    );
                    continue;
                }
            };

            // Get value at index
            let value = match f64_series.get(index) {
                Some(value) => value,
                None => {
                    debug_once!(
                        "Attribute '{}' at index {} is None, skipping",
                        col_name,
                        index
                    );
                    continue;
                }
            };

            // Skip NaN values
            if value.is_nan() {
                debug_once!(
                    "Attribute '{}' at index {} is NaN, skipping",
                    col_name,
                    index
                );
                continue;
            }

            attributes.insert(col_name.to_string(), value);
        }

        Ok(Point {
            x,
            y,
            z,
            attributes,
        })
    }

    /// Convert to a vector of Points
    pub fn to_points(&self) -> Result<Vec<Point<f64>>, PolarsError> {
        let mut points = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            points.push(self.get_point(i)?);
        }
        Ok(points)
    }
}

impl Default for TablePointCloud {
    fn default() -> Self {
        Self::new().expect("Failed to create default TablePointCloud")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    #[test]
    fn test_table_point_cloud_new() {
        let cloud = TablePointCloud::new().unwrap();
        assert_eq!(cloud.len(), 0);
        assert!(cloud.is_empty());
    }

    #[test]
    fn test_table_point_cloud_from_xyz() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let z = vec![7.0, 8.0, 9.0];

        let cloud = TablePointCloud::from_xyz(x, y, z).unwrap();
        assert_eq!(cloud.len(), 3);
        assert!(!cloud.is_empty());

        let x_coords = cloud.x().unwrap();
        assert_eq!(x_coords, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_table_point_cloud_from_points() {
        let mut points = vec![Point::new(1.0, 2.0, 3.0), Point::new(4.0, 5.0, 6.0)];

        points[0].set_attribute("intensity".to_string(), 100.0);
        points[1].set_attribute("intensity".to_string(), 200.0);

        let cloud = TablePointCloud::from_points(points).unwrap();
        assert_eq!(cloud.len(), 2);

        let point0 = cloud.get_point(0).unwrap();
        assert_eq!(point0.x, 1.0);
        assert_eq!(point0.y, 2.0);
        assert_eq!(point0.z, 3.0);
        assert_eq!(point0.get_attribute("intensity"), Some(100.0));
    }

    #[test]
    fn test_table_point_cloud_add_attribute() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let z = vec![5.0, 6.0];

        let mut cloud = TablePointCloud::from_xyz(x, y, z).unwrap();
        cloud
            .add_attribute("intensity", vec![100.0, 200.0])
            .unwrap();

        let point0 = cloud.get_point(0).unwrap();
        assert_eq!(point0.get_attribute("intensity"), Some(100.0));
    }

    #[test]
    fn test_table_point_cloud_to_points() {
        let points = vec![Point::new(1.0, 2.0, 3.0), Point::new(4.0, 5.0, 6.0)];

        let cloud = TablePointCloud::from_points(points.clone()).unwrap();
        let recovered_points = cloud.to_points().unwrap();

        assert_eq!(recovered_points.len(), 2);
        assert_eq!(recovered_points[0].x, 1.0);
        assert_eq!(recovered_points[1].x, 4.0);
    }

    #[test]
    fn test_point_cloud_mismatched_lengths() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0, 5.0]; // Different length
        let z = vec![6.0, 7.0];

        let result = TablePointCloud::from_xyz(x, y, z);
        assert!(result.is_err());
    }

    #[test]
    fn test_point_cloud_transform_identity() {
        // Test with identity matrix (should not change points)
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let z = vec![7.0, 8.0, 9.0];

        let cloud = TablePointCloud::from_xyz(x.clone(), y.clone(), z.clone()).unwrap();
        let identity = Matrix4::identity();
        let transformed = cloud.transform(&identity).unwrap();

        let tx = transformed.x().unwrap();
        let ty = transformed.y().unwrap();
        let tz = transformed.z().unwrap();

        assert_eq!(tx, x);
        assert_eq!(ty, y);
        assert_eq!(tz, z);
    }

    #[test]
    fn test_point_cloud_transform_translation() {
        // Test translation by (10, 20, 30)
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let z = vec![5.0, 6.0];

        let cloud = TablePointCloud::from_xyz(x, y, z).unwrap();

        // Create translation matrix
        let mut transform = Matrix4::identity();
        transform[(0, 3)] = 10.0; // tx
        transform[(1, 3)] = 20.0; // ty
        transform[(2, 3)] = 30.0; // tz

        let transformed = cloud.transform(&transform).unwrap();

        let tx = transformed.x().unwrap();
        let ty = transformed.y().unwrap();
        let tz = transformed.z().unwrap();

        assert_eq!(tx, vec![11.0, 12.0]);
        assert_eq!(ty, vec![23.0, 24.0]);
        assert_eq!(tz, vec![35.0, 36.0]);
    }

    #[test]
    fn test_point_cloud_transform_preserves_attributes() {
        // Test that transformation preserves attributes
        let mut points = vec![Point::new(1.0, 2.0, 3.0)];
        points[0].set_attribute("intensity".to_string(), 100.0);

        let cloud = TablePointCloud::from_points(points).unwrap();

        let transform = Matrix4::identity();
        let transformed = cloud.transform(&transform).unwrap();

        let point = transformed.get_point(0).unwrap();
        assert_eq!(point.get_attribute("intensity"), Some(100.0));
    }
}
