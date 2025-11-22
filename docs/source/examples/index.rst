Examples
========

Point Creation and Manipulation
--------------------------------

.. code-block:: python

   import pcl_rustic

   # Create a simple point
   point = pcl_rustic.Point(1.0, 2.0, 3.0)
   print(f"Point coordinates: x={point.x}, y={point.y}, z={point.z}")

   # Create a point with attributes
   point_with_attrs = pcl_rustic.Point(4.0, 5.0, 6.0, {
       "intensity": 255.0,
       "red": 100.0,
       "green": 150.0
   })

   # Modify attributes
   point.set_attribute("intensity", 200.0)
   intensity = point.get_attribute("intensity")
   print(f"Intensity: {intensity}")

Point Cloud Creation
--------------------

From Coordinate Lists
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pcl_rustic

   # Create from separate coordinate lists
   x = [1.0, 2.0, 3.0, 4.0, 5.0]
   y = [10.0, 20.0, 30.0, 40.0, 50.0]
   z = [100.0, 200.0, 300.0, 400.0, 500.0]

   cloud = pcl_rustic.TablePointCloud.from_xyz(x, y, z)
   print(f"Cloud has {len(cloud)} points")

From Points
~~~~~~~~~~~

.. code-block:: python

   import pcl_rustic

   # Create from a list of Point objects
   points = [
       pcl_rustic.Point(1.0, 0.0, 0.0, {"intensity": 100.0}),
       pcl_rustic.Point(0.0, 1.0, 0.0, {"intensity": 200.0}),
       pcl_rustic.Point(0.0, 0.0, 1.0, {"intensity": 300.0}),
   ]

   cloud = pcl_rustic.TablePointCloud.from_points(points)

Accessing Point Cloud Data
---------------------------

.. code-block:: python

   import pcl_rustic

   cloud = pcl_rustic.TablePointCloud.from_xyz(
       [1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0],
       [7.0, 8.0, 9.0]
   )

   # Get coordinate lists
   x_coords = cloud.x()
   y_coords = cloud.y()
   z_coords = cloud.z()

   # Get individual points
   point = cloud.get_point(0)
   print(f"First point: ({point.x}, {point.y}, {point.z})")

   # Convert to list of points
   all_points = cloud.to_points()

Managing Attributes
-------------------

.. code-block:: python

   import pcl_rustic

   cloud = pcl_rustic.TablePointCloud.from_xyz(
       [1.0, 2.0],
       [3.0, 4.0],
       [5.0, 6.0]
   )

   # Add an attribute column
   cloud.add_attribute("intensity", [100.0, 200.0])

   # Retrieve point with attributes
   point = cloud.get_point(0)
   print(f"Intensity: {point.get_attribute('intensity')}")

Spatial Transformations
-----------------------

Translation
~~~~~~~~~~~

.. code-block:: python

   import pcl_rustic

   cloud = pcl_rustic.TablePointCloud.from_xyz(
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0]
   )

   # Translation matrix (translate by 10, 20, 30)
   # Matrix format: row-major order [m00, m01, m02, m03, m10, m11, m12, m13, ...]
   translation = [
       1.0, 0.0, 0.0, 10.0,
       0.0, 1.0, 0.0, 20.0,
       0.0, 0.0, 1.0, 30.0,
       0.0, 0.0, 0.0, 1.0
   ]

   transformed = cloud.transform(translation)

   # Check results
   for i in range(len(transformed)):
       p = transformed.get_point(i)
       print(f"Point {i}: ({p.x}, {p.y}, {p.z})")

Rotation
~~~~~~~~

.. code-block:: python

   import pcl_rustic

   cloud = pcl_rustic.TablePointCloud.from_xyz(
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0]
   )

   # 90-degree rotation around Z-axis
   rotation = [
       0.0, -1.0, 0.0, 0.0,
       1.0,  0.0, 0.0, 0.0,
       0.0,  0.0, 1.0, 0.0,
       0.0,  0.0, 0.0, 1.0
   ]

   rotated = cloud.transform(rotation)

   # Points are now rotated
   for i in range(len(rotated)):
       p = rotated.get_point(i)
       print(f"Point {i}: ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")

Combined Transformation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pcl_rustic

   # You can combine rotation and translation in a single matrix
   combined = [
       0.0, -1.0, 0.0, 10.0,  # Rotate and translate
       1.0,  0.0, 0.0, 20.0,
       0.0,  0.0, 1.0, 30.0,
       0.0,  0.0, 0.0,  1.0
   ]

   cloud = pcl_rustic.TablePointCloud.from_xyz([1, 0, 0], [0, 1, 0], [0, 0, 1])
   transformed = cloud.transform(combined)
