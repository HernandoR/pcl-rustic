"""
pcl_rustic._core - Rust实现的核心库

类型注解和API规范
"""

from typing import Optional, Dict, List

class PointCloud:
    """高性能点云类

    属性：
        - xyz: [M,3]的三维坐标
        - intensity: 可选的强度值[M]
        - rgb: 可选的RGB颜色[M,3]
        - attributes: 自定义属性字典
    """

    def __new__(cls) -> "PointCloud": ...
    @staticmethod
    def from_xyz(xyz: List[List[float]]) -> "PointCloud": ...
    def point_count(self) -> int: ...
    def get_xyz(self) -> List[List[float]]: ...
    def has_intensity(self) -> bool: ...
    def has_rgb(self) -> bool: ...
    def get_intensity(self) -> Optional[List[float]]: ...
    def get_rgb(self) -> Optional[List[List[int]]]: ...
    def set_intensity(self, intensity: List[float]) -> None: ...
    def set_rgb(self, rgb: List[List[int]]) -> None: ...
    def add_attribute(self, name: str, data: List[float]) -> None: ...
    def set_attribute(self, name: str, data: List[float]) -> None: ...
    def attribute_names(self) -> List[str]: ...
    def get_attribute(self, name: str) -> Optional[List[float]]: ...
    def remove_attribute(self, name: str) -> None: ...
    def transform(self, matrix: List[List[float]]) -> "PointCloud": ...
    def rigid_transform(
        self, rotation: List[List[float]], translation: List[float]
    ) -> "PointCloud": ...
    def voxel_downsample(self, voxel_size: float, strategy: int) -> "PointCloud": ...
    @staticmethod
    def from_las(path: str) -> "PointCloud": ...
    def to_las(self, path: str, compress: bool = False) -> None: ...
    @staticmethod
    def from_csv(
        path: str,
        delimiter: int = ord(b","),
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        intensity: str | None = None,
        rgb_r: str | None = None,
        rgb_g: str | None = None,
        rgb_b: str | None = None,
    ) -> "PointCloud": ...
    @staticmethod
    def from_parquet(
        path: str,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        intensity: str | None = None,
        rgb_r: str | None = None,
        rgb_g: str | None = None,
        rgb_b: str | None = None,
    ) -> "PointCloud": ...
    def to_csv(
        self,
        path: str,
        delimiter: int = ord(b","),
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        intensity: str | None = None,
        rgb_r: str | None = None,
        rgb_g: str | None = None,
        rgb_b: str | None = None,
    ) -> None: ...
    def to_parquet(
        self,
        path: str,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        intensity: str | None = None,
        rgb_r: str | None = None,
        rgb_g: str | None = None,
        rgb_b: str | None = None,
    ) -> None: ...
    @staticmethod
    def load_from_file(
        path: str,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        intensity: str | None = None,
        rgb_r: str | None = None,
        rgb_g: str | None = None,
        rgb_b: str | None = None,
    ) -> "PointCloud": ...
    def save_to_file(
        self,
        path: str,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        intensity: str | None = None,
        rgb_r: str | None = None,
        rgb_g: str | None = None,
        rgb_b: str | None = None,
    ) -> None: ...
    def memory_usage(self) -> int: ...
    def to_dict(self) -> Dict[str, any]: ...
    def clone(self) -> "PointCloud": ...
    def __repr__(self) -> str: ...

class DownsampleStrategy:
    """下采样策略枚举"""

    RANDOM: int
    """随机采样策略"""

    CENTROID: int
    """重心采样策略（选择最接近体素中心的点）"""
