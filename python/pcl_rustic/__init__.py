from pcl_rustic._core import hello_from_bind, PyPoint, PyTablePointCloud

# Export the classes with cleaner names
Point = PyPoint
TablePointCloud = PyTablePointCloud

__all__ = ['hello_from_bind', 'Point', 'TablePointCloud']


def main() -> None:
    print(hello_from_bind())

