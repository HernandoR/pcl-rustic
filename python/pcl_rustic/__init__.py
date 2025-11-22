"""
PCL-Rustic: Point Cloud Library in Rust with Python bindings.

This module provides efficient point cloud data structures and operations
using Rust backend with polars DataFrame for columnar storage.

Classes:
    Point: A 3D point with coordinates and optional attributes.
    TablePointCloud: A columnar point cloud using polars DataFrame backend.

Example:
    >>> import pcl_rustic
    >>> # Create a point
    >>> point = pcl_rustic.Point(1.0, 2.0, 3.0)
    >>> # Create a point cloud
    >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
    >>> # Transform the cloud
    >>> matrix = [1,0,0,10, 0,1,0,20, 0,0,1,30, 0,0,0,1]
    >>> transformed = cloud.transform(matrix)
"""

from pcl_rustic._core import hello_from_bind, PyPoint, PyTablePointCloud

# Export the classes with cleaner names
Point = PyPoint
TablePointCloud = PyTablePointCloud

__version__ = "0.1.0"
__all__ = ['hello_from_bind', 'Point', 'TablePointCloud', '__version__']


def main() -> None:
    """Entry point for the pcl-rustic CLI."""
    print(hello_from_bind())

