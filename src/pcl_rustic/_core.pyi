"""
pcl_rustic._core - Rust实现的核心库

类型注解和API规范（基于NumPy数组，dtype=float32）
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

class PointCloud:
    """高性能点云类

    属性：
        - xyz: [N,3]的三维坐标 (float32)
        - intensity: 可选的强度值 [N] (float32)
        - rgb: 可选的RGB颜色通道 [N] (float32 输入, uint8 输出)
        - attributes: 自定义属性字典 (float32)
    """

    def __new__(cls) -> "PointCloud": ...
    @staticmethod
    def from_xyz(xyz: NDArray[np.float32]) -> "PointCloud": ...
    @staticmethod
    def from_xyz_intensity(
        xyz: NDArray[np.float32], intensity: NDArray[np.float32]
    ) -> "PointCloud": ...
    @staticmethod
    def from_xyz_rgb(
        xyz: NDArray[np.float32],
        r: NDArray[np.float32],
        g: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> "PointCloud": ...
    @staticmethod
    def from_xyz_intensity_rgb(
        xyz: NDArray[np.float32],
        intensity: NDArray[np.float32],
        r: NDArray[np.float32],
        g: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> "PointCloud": ...
    @staticmethod
    def from_dict(data: Dict[str, NDArray[np.float32]]) -> "PointCloud": ...
    def point_count(self) -> int: ...
    def get_xyz(self) -> NDArray[np.float32]: ...
    def has_intensity(self) -> bool: ...
    def has_rgb(self) -> bool: ...
    def get_intensity(self) -> Optional[NDArray[np.float32]]: ...
    def get_rgb(
        self,
    ) -> Optional[Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]]: ...
    def set_intensity(self, intensity: NDArray[np.float32]) -> None: ...
    def set_rgb(
        self,
        r: NDArray[np.float32],
        g: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> None: ...
    def add_attribute(self, name: str, data: NDArray[np.float32]) -> None: ...
    def set_attribute(self, name: str, data: NDArray[np.float32]) -> None: ...
    def attribute_names(self) -> List[str]: ...
    def get_attribute(self, name: str) -> Optional[NDArray[np.float32]]: ...
    def remove_attribute(self, name: str) -> None: ...
    def clear_attributes(self) -> None: ...
    def set_all_attributes(self, attributes: Dict[str, List[float]]) -> None: ...
    def has_attributes(self, names: List[str]) -> bool: ...
    def attribute_info(self) -> List[Tuple[str, int]]: ...
    def remove_intensity(self) -> None: ...
    def remove_rgb(self) -> None: ...
    @staticmethod
    def delete_file(path: str) -> None: ...
    def transform(self, matrix: NDArray[np.float32]) -> "PointCloud": ...
    def rigid_transform(
        self, rotation: NDArray[np.float32], translation: NDArray[np.float32]
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
    def to_dict(self) -> Dict[str, NDArray[np.float32]]: ...
    def clone(self) -> "PointCloud": ...
    def __repr__(self) -> str: ...

class DownsampleStrategy:
    """下采样策略枚举"""

    RANDOM: int
    """随机采样策略"""

    CENTROID: int
    """重心采样策略（最接近体素中心的点）"""
