# Python API Examples

This document provides examples of using the pcl-rustic Python API.

## Installation

Build and install the package:

```bash
maturin build --release
pip install target/wheels/pcl_rustic-*.whl
```

## Point API

### Creating Points

```python
import pcl_rustic

# Simple point with just coordinates
point = pcl_rustic.Point(1.0, 2.0, 3.0)

# Point with attributes
point = pcl_rustic.Point(1.0, 2.0, 3.0, {
    "intensity": 255.0,
    "red": 100.0,
    "green": 150.0
})
```

### Accessing Point Data

```python
# Access coordinates
x = point.x
y = point.y
z = point.z

# Get all attributes
attrs = point.attributes

# Get specific attribute
intensity = point.get_attribute("intensity")

# Set attribute
point.set_attribute("custom_field", 42.0)
```

## TablePointCloud API

### Creating Point Clouds

```python
import pcl_rustic

# From coordinate lists
x = [1.0, 2.0, 3.0]
y = [4.0, 5.0, 6.0]
z = [7.0, 8.0, 9.0]
cloud = pcl_rustic.TablePointCloud.from_xyz(x, y, z)

# From points
points = [
    pcl_rustic.Point(1.0, 0.0, 0.0, {"intensity": 100.0}),
    pcl_rustic.Point(0.0, 1.0, 0.0, {"intensity": 200.0}),
]
cloud = pcl_rustic.TablePointCloud.from_points(points)

# Empty cloud
cloud = pcl_rustic.TablePointCloud()
```

### Accessing Cloud Data

```python
# Get number of points
num_points = len(cloud)

# Check if empty
is_empty = cloud.is_empty()

# Get coordinates as lists
x_coords = cloud.x()
y_coords = cloud.y()
z_coords = cloud.z()

# Get individual point
point = cloud.get_point(0)

# Convert to list of points
all_points = cloud.to_points()
```

### Managing Attributes

```python
# Add attribute column
cloud.add_attribute("intensity", [10.0, 20.0, 30.0])

# Attributes are preserved when converting
points = cloud.to_points()
intensity = points[0].get_attribute("intensity")
```

## Transformations

The `transform()` method applies a 4x4 homogeneous transformation matrix to all points in the cloud.

### Matrix Format

The transformation matrix is provided as a flat list of 16 values in **row-major order**:

```python
matrix = [
    m00, m01, m02, m03,  # row 0
    m10, m11, m12, m13,  # row 1
    m20, m21, m22, m23,  # row 2
    m30, m31, m32, m33   # row 3
]
```

### Translation Example

Translate all points by (tx, ty, tz):

```python
# Translation matrix: translate by (10, 20, 30)
translation = [
    1.0, 0.0, 0.0, 10.0,
    0.0, 1.0, 0.0, 20.0,
    0.0, 0.0, 1.0, 30.0,
    0.0, 0.0, 0.0, 1.0
]

transformed = cloud.transform(translation)
```

### Rotation Example

Rotate 90 degrees around the Z-axis:

```python
import math

# 90-degree rotation around Z
rotation = [
    0.0, -1.0, 0.0, 0.0,
    1.0,  0.0, 0.0, 0.0,
    0.0,  0.0, 1.0, 0.0,
    0.0,  0.0, 0.0, 1.0
]

rotated = cloud.transform(rotation)
```

### Combined Transformation

You can combine rotation and translation:

```python
# Rotate around Z and translate
combined = [
    0.0, -1.0, 0.0, 10.0,
    1.0,  0.0, 0.0, 20.0,
    0.0,  0.0, 1.0, 30.0,
    0.0,  0.0, 0.0,  1.0
]

transformed = cloud.transform(combined)
```

## Complete Example

```python
import pcl_rustic

# Create a point cloud
points = [
    pcl_rustic.Point(1.0, 0.0, 0.0, {"intensity": 100.0}),
    pcl_rustic.Point(0.0, 1.0, 0.0, {"intensity": 200.0}),
    pcl_rustic.Point(0.0, 0.0, 1.0, {"intensity": 300.0}),
]
cloud = pcl_rustic.TablePointCloud.from_points(points)

print(f"Original cloud has {len(cloud)} points")

# Apply transformation (translate by 10 in all directions)
transform_matrix = [
    1.0, 0.0, 0.0, 10.0,
    0.0, 1.0, 0.0, 10.0,
    0.0, 0.0, 1.0, 10.0,
    0.0, 0.0, 0.0,  1.0
]

transformed = cloud.transform(transform_matrix)

# Get transformed points
for i in range(len(transformed)):
    p = transformed.get_point(i)
    print(f"Point {i}: ({p.x}, {p.y}, {p.z}), intensity: {p.get_attribute('intensity')}")
```

## Notes

- All coordinates are stored as `f64` (Python `float`)
- Attributes are key-value pairs where keys are strings and values are `f64`
- Transformations use homogeneous coordinates internally `[x, y, z, 1]`
- The transformation follows the convention: `P_b = T_a2b @ P_a` (right multiplication)
- Attribute columns are preserved during transformations
