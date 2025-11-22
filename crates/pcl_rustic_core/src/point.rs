//! Point data structure module.
//!
//! This module provides the [`Point`] struct, a generic 3D point with optional attributes.

use nalgebra::Scalar;
use std::collections::HashMap;

/// A 3D point with required coordinates and optional attributes.
///
/// The point is generic over numeric type `T` that supports matrix operations.
/// By default, it uses `f64` for coordinates.
///
/// # Type Parameters
///
/// * `T` - The numeric type for coordinates. Must implement `Scalar + Copy`.
///   Common types include `f64`, `f32`, and integer types.
///
/// # Fields
///
/// * `x` - X coordinate
/// * `y` - Y coordinate  
/// * `z` - Z coordinate
/// * `attributes` - Custom attributes stored as key-value pairs (key: String, value: f64)
///
/// # Examples
///
/// ```
/// use pcl_rustic_core::Point;
/// use std::collections::HashMap;
///
/// // Create a simple point with f64 coordinates
/// let point = Point::new(1.0, 2.0, 3.0);
/// assert_eq!(point.x, 1.0);
/// assert_eq!(point.y, 2.0);
/// assert_eq!(point.z, 3.0);
///
/// // Create a point with f32 coordinates
/// let point_f32: Point<f32> = Point::new(1.0, 2.0, 3.0);
///
/// // Create a point with attributes
/// let mut attrs = HashMap::new();
/// attrs.insert("intensity".to_string(), 255.0);
/// let point = Point::with_attributes(1.0, 2.0, 3.0, attrs);
/// assert_eq!(point.get_attribute("intensity"), Some(255.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Point<T = f64>
where
    T: Scalar + Copy,
{
    pub x: T,
    pub y: T,
    pub z: T,
    pub attributes: HashMap<String, f64>,
}

impl<T> Point<T>
where
    T: Scalar + Copy,
{
    /// Create a new point with x, y, z coordinates.
    ///
    /// The point is created with no attributes.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    /// * `z` - Z coordinate
    ///
    /// # Examples
    ///
    /// ```
    /// use pcl_rustic_core::Point;
    ///
    /// let point = Point::new(1.0, 2.0, 3.0);
    /// assert!(point.attributes.is_empty());
    /// ```
    pub fn new(x: T, y: T, z: T) -> Self {
        Point {
            x,
            y,
            z,
            attributes: HashMap::new(),
        }
    }

    /// Create a new point with x, y, z coordinates and attributes.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    /// * `z` - Z coordinate
    /// * `attributes` - HashMap of custom attributes
    ///
    /// # Examples
    ///
    /// ```
    /// use pcl_rustic_core::Point;
    /// use std::collections::HashMap;
    ///
    /// let mut attrs = HashMap::new();
    /// attrs.insert("intensity".to_string(), 100.0);
    /// attrs.insert("red".to_string(), 255.0);
    ///
    /// let point = Point::with_attributes(1.0, 2.0, 3.0, attrs);
    /// assert_eq!(point.attributes.len(), 2);
    /// ```
    pub fn with_attributes(x: T, y: T, z: T, attributes: HashMap<String, f64>) -> Self {
        Point {
            x,
            y,
            z,
            attributes,
        }
    }

    /// Add or update an attribute.
    ///
    /// If the attribute already exists, its value is updated.
    ///
    /// # Arguments
    ///
    /// * `key` - Attribute name
    /// * `value` - Attribute value
    ///
    /// # Examples
    ///
    /// ```
    /// use pcl_rustic_core::Point;
    ///
    /// let mut point = Point::new(1.0, 2.0, 3.0);
    /// point.set_attribute("intensity".to_string(), 150.0);
    /// assert_eq!(point.get_attribute("intensity"), Some(150.0));
    /// ```
    pub fn set_attribute(&mut self, key: String, value: f64) {
        self.attributes.insert(key, value);
    }

    /// Get an attribute value.
    ///
    /// # Arguments
    ///
    /// * `key` - Attribute name to retrieve
    ///
    /// # Returns
    ///
    /// * `Some(value)` if the attribute exists
    /// * `None` if the attribute doesn't exist
    ///
    /// # Examples
    ///
    /// ```
    /// use pcl_rustic_core::Point;
    ///
    /// let mut point = Point::new(1.0, 2.0, 3.0);
    /// point.set_attribute("intensity".to_string(), 150.0);
    ///
    /// assert_eq!(point.get_attribute("intensity"), Some(150.0));
    /// assert_eq!(point.get_attribute("nonexistent"), None);
    /// ```
    pub fn get_attribute(&self, key: &str) -> Option<f64> {
        self.attributes.get(key).copied()
    }
}

impl<T> Default for Point<T>
where
    T: Scalar + Copy + Default,
{
    /// Create a default point at the origin (0, 0, 0) with no attributes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pcl_rustic_core::Point;
    ///
    /// let point: Point<f64> = Point::default();
    /// assert_eq!(point.x, 0.0);
    /// assert_eq!(point.y, 0.0);
    /// assert_eq!(point.z, 0.0);
    /// assert!(point.attributes.is_empty());
    /// ```
    fn default() -> Self {
        Point::new(T::default(), T::default(), T::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let point = Point::new(1.0, 2.0, 3.0);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert!(point.attributes.is_empty());
    }

    #[test]
    fn test_point_default() {
        let point: Point<f64> = Point::default();
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);
        assert!(point.attributes.is_empty());
    }

    #[test]
    fn test_point_with_attributes() {
        let mut attrs = HashMap::new();
        attrs.insert("intensity".to_string(), 100.0);
        attrs.insert("red".to_string(), 255.0);

        let point = Point::with_attributes(1.0, 2.0, 3.0, attrs);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.get_attribute("intensity"), Some(100.0));
        assert_eq!(point.get_attribute("red"), Some(255.0));
    }

    #[test]
    fn test_point_set_get_attribute() {
        let mut point = Point::new(1.0, 2.0, 3.0);
        point.set_attribute("intensity".to_string(), 150.0);
        assert_eq!(point.get_attribute("intensity"), Some(150.0));
        assert_eq!(point.get_attribute("nonexistent"), None);
    }

    #[test]
    fn test_point_generic_types() {
        // Test with f32
        let point_f32 = Point::<f32>::new(1.0, 2.0, 3.0);
        assert_eq!(point_f32.x, 1.0f32);

        // Test with i32
        let point_i32 = Point::<i32>::new(1, 2, 3);
        assert_eq!(point_i32.x, 1);
    }
}
