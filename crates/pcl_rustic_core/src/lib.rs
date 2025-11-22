//! PCL-Rustic Core Library
//!
//! This crate provides the core data structures and functionality for point cloud processing.
//!
//! # Overview
//!
//! The library provides two main types:
//! - [`Point`]: A generic 3D point with optional attributes
//! - [`TablePointCloud`]: A columnar point cloud using polars DataFrame
//!
//! # Examples
//!
//! ```
//! use pcl_rustic_core::{Point, TablePointCloud};
//!
//! // Create a point
//! let point = Point::new(1.0, 2.0, 3.0);
//!
//! // Create a point cloud
//! let cloud = TablePointCloud::from_xyz(
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0]
//! ).unwrap();
//! ```

mod point;
mod point_cloud;

pub use point::Point;
pub use point_cloud::TablePointCloud;

/// Returns a greeting message from the core library.
///
/// This is a simple test function to verify the library is working correctly.
///
/// # Examples
///
/// ```
/// use pcl_rustic_core::hello_from_core;
///
/// let message = hello_from_core();
/// assert_eq!(message, "Hello from pcl_rustic core!");
/// ```
pub fn hello_from_core() -> String {
    "Hello from pcl_rustic core!".to_string()
}
