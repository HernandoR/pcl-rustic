"""
PCL-Rustic: Point Cloud Library in Rust with Python bindings.

This module provides efficient point cloud data structures and operations
using Rust backend with polars DataFrame for columnar storage.
"""

from typing import Dict, List, Optional, Tuple

__version__: str = "0.1.0"
__all__: List[str] = ["Point", "TablePointCloud", "hello_from_bind"]

def hello_from_bind() -> str:
    """
    Test function that returns a greeting from the Rust core.
    
    Returns:
        str: A greeting message from the Rust implementation.
    
    Example:
        >>> import pcl_rustic
        >>> pcl_rustic.hello_from_bind()
        'Hello from pcl_rustic core!'
    """
    ...

class Point:
    """
    A 3D point with coordinates and optional attributes.
    
    Points are the fundamental building blocks of point clouds. Each point
    has required x, y, z coordinates and can have optional custom attributes
    stored as key-value pairs.
    
    Attributes:
        x (float): X coordinate of the point.
        y (float): Y coordinate of the point.
        z (float): Z coordinate of the point.
        attributes (Dict[str, float]): Dictionary of custom attributes.
    
    Example:
        >>> import pcl_rustic
        >>> # Create a simple point
        >>> point = pcl_rustic.Point(1.0, 2.0, 3.0)
        >>> print(point.x, point.y, point.z)
        1.0 2.0 3.0
        >>> 
        >>> # Create a point with attributes
        >>> point = pcl_rustic.Point(4.0, 5.0, 6.0, {"intensity": 255.0})
        >>> print(point.get_attribute("intensity"))
        255.0
    """
    
    x: float
    y: float
    z: float
    attributes: Dict[str, float]
    
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        attributes: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Create a new 3D point.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate.
            attributes: Optional dictionary of custom attributes (key: str, value: float).
        
        Example:
            >>> point = pcl_rustic.Point(1.0, 2.0, 3.0)
            >>> point_with_attrs = pcl_rustic.Point(1.0, 2.0, 3.0, {"intensity": 100.0})
        """
        ...
    
    def set_attribute(self, key: str, value: float) -> None:
        """
        Set or update an attribute value.
        
        Args:
            key: Attribute name.
            value: Attribute value.
        
        Example:
            >>> point = pcl_rustic.Point(1.0, 2.0, 3.0)
            >>> point.set_attribute("intensity", 200.0)
        """
        ...
    
    def get_attribute(self, key: str) -> Optional[float]:
        """
        Get an attribute value.
        
        Args:
            key: Attribute name to retrieve.
        
        Returns:
            The attribute value if it exists, None otherwise.
        
        Example:
            >>> point = pcl_rustic.Point(1.0, 2.0, 3.0, {"intensity": 100.0})
            >>> intensity = point.get_attribute("intensity")
            >>> print(intensity)
            100.0
        """
        ...
    
    def __repr__(self) -> str:
        """Return a string representation of the point."""
        ...

class TablePointCloud:
    """
    A columnar point cloud using polars DataFrame backend.
    
    TablePointCloud stores point cloud data in a columnar format where each
    attribute (including x, y, z coordinates) is stored as a separate column.
    This provides efficient memory usage and fast operations on point cloud data.
    
    All point clouds are assumed to be 3D and must have x, y, z columns.
    Additional attributes can be added as additional columns.
    
    Example:
        >>> import pcl_rustic
        >>> # Create from coordinate lists
        >>> cloud = pcl_rustic.TablePointCloud.from_xyz(
        ...     [1.0, 2.0, 3.0],
        ...     [4.0, 5.0, 6.0],
        ...     [7.0, 8.0, 9.0]
        ... )
        >>> print(len(cloud))
        3
    """
    
    def __init__(self) -> None:
        """
        Create a new empty TablePointCloud.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud()
            >>> print(len(cloud))
            0
        """
        ...
    
    @staticmethod
    def from_xyz(x: List[float], y: List[float], z: List[float]) -> "TablePointCloud":
        """
        Create a TablePointCloud from coordinate lists.
        
        Args:
            x: List of x coordinates.
            y: List of y coordinates.
            z: List of z coordinates.
        
        Returns:
            A new TablePointCloud with the specified coordinates.
        
        Raises:
            ValueError: If coordinate lists have different lengths.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz(
            ...     [1.0, 2.0],
            ...     [3.0, 4.0],
            ...     [5.0, 6.0]
            ... )
        """
        ...
    
    @staticmethod
    def from_points(points: List[Point]) -> "TablePointCloud":
        """
        Create a TablePointCloud from a list of Point objects.
        
        Attributes from the points are automatically extracted and stored
        as columns in the cloud. If points have different attributes,
        missing values are filled with NaN.
        
        Args:
            points: List of Point objects.
        
        Returns:
            A new TablePointCloud containing all points and their attributes.
        
        Example:
            >>> points = [
            ...     pcl_rustic.Point(1.0, 0.0, 0.0, {"intensity": 100.0}),
            ...     pcl_rustic.Point(0.0, 1.0, 0.0, {"intensity": 200.0}),
            ... ]
            >>> cloud = pcl_rustic.TablePointCloud.from_points(points)
        """
        ...
    
    def __len__(self) -> int:
        """
        Get the number of points in the cloud.
        
        Returns:
            The number of points.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> len(cloud)
            2
        """
        ...
    
    def is_empty(self) -> bool:
        """
        Check if the point cloud is empty.
        
        Returns:
            True if the cloud has no points, False otherwise.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud()
            >>> cloud.is_empty()
            True
        """
        ...
    
    def x(self) -> List[float]:
        """
        Get all x coordinates as a list.
        
        Returns:
            List of x coordinates.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> cloud.x()
            [1.0, 2.0]
        """
        ...
    
    def y(self) -> List[float]:
        """
        Get all y coordinates as a list.
        
        Returns:
            List of y coordinates.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> cloud.y()
            [3.0, 4.0]
        """
        ...
    
    def z(self) -> List[float]:
        """
        Get all z coordinates as a list.
        
        Returns:
            List of z coordinates.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> cloud.z()
            [5.0, 6.0]
        """
        ...
    
    def add_attribute(self, name: str, values: List[float]) -> None:
        """
        Add an attribute column to the point cloud.
        
        Args:
            name: Name of the attribute.
            values: List of attribute values, must match the number of points.
        
        Raises:
            ValueError: If values list length doesn't match the number of points.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> cloud.add_attribute("intensity", [100.0, 200.0])
        """
        ...
    
    def get_point(self, index: int) -> Point:
        """
        Get a Point at a specific index.
        
        Args:
            index: Index of the point to retrieve (0-based).
        
        Returns:
            The Point at the specified index with all its attributes.
        
        Raises:
            ValueError: If index is out of bounds.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> point = cloud.get_point(0)
            >>> print(point.x, point.y, point.z)
            1.0 3.0 5.0
        """
        ...
    
    def to_points(self) -> List[Point]:
        """
        Convert the point cloud to a list of Point objects.
        
        Returns:
            List of all points in the cloud with their attributes.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 2], [3, 4], [5, 6])
            >>> points = cloud.to_points()
            >>> len(points)
            2
        """
        ...
    
    def transform(self, matrix: List[float]) -> "TablePointCloud":
        """
        Transform the point cloud using a 4x4 homogeneous transformation matrix.
        
        The transformation applies the matrix to each point using homogeneous
        coordinates [x, y, z, 1]. The matrix should be provided as a flat list
        of 16 values in row-major order.
        
        Performs right multiplication: P_b = T_a2b @ P_a
        
        Args:
            matrix: Flat list of 16 values representing a 4x4 transformation matrix
                   in row-major order: [m00, m01, m02, m03, m10, m11, m12, m13, ...]
        
        Returns:
            A new transformed TablePointCloud. Attributes are preserved.
        
        Raises:
            ValueError: If matrix doesn't have exactly 16 elements.
        
        Example:
            >>> cloud = pcl_rustic.TablePointCloud.from_xyz([1, 0], [0, 1], [0, 0])
            >>> # Translation matrix: translate by (10, 20, 30)
            >>> translation = [
            ...     1.0, 0.0, 0.0, 10.0,
            ...     0.0, 1.0, 0.0, 20.0,
            ...     0.0, 0.0, 1.0, 30.0,
            ...     0.0, 0.0, 0.0, 1.0
            ... ]
            >>> transformed = cloud.transform(translation)
        """
        ...
    
    def __repr__(self) -> str:
        """Return a string representation of the point cloud."""
        ...
