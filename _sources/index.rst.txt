Welcome to pcl-rustic's documentation!
=======================================

**pcl-rustic** is a Rust-based point cloud processing library with Python bindings,
providing efficient operations on 3D point cloud data using polars DataFrame backend.

Features
--------

* **Generic Point Structure**: Support for multiple numeric types (f32, f64, i32, etc.)
* **Columnar Storage**: Efficient point cloud storage using polars DataFrame
* **Spatial Transformations**: 4x4 homogeneous transformation matrix support
* **Python API**: Complete Python bindings with PyO3
* **Flexible Attributes**: Custom attributes stored as key-value pairs

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install pcl-rustic

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import pcl_rustic

   # Create a point
   point = pcl_rustic.Point(1.0, 2.0, 3.0)

   # Create a point cloud
   cloud = pcl_rustic.TablePointCloud.from_xyz(
       [1.0, 2.0, 3.0],
       [4.0, 5.0, 6.0],
       [7.0, 8.0, 9.0]
   )

   # Transform the cloud
   translation_matrix = [
       1.0, 0.0, 0.0, 10.0,
       0.0, 1.0, 0.0, 20.0,
       0.0, 0.0, 1.0, 30.0,
       0.0, 0.0, 0.0, 1.0
   ]
   transformed = cloud.transform(translation_matrix)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
