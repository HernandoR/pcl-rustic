#!/usr/bin/env python3
"""pcl-rustic usage examples.

Demonstrates construction, dimension access, the laspy-aligned dtype
contract, geometry, device selection, file I/O, and optional torch interop.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from pcl_rustic import DownsampleStrategy, PointCloud, available_devices, default_device


def example_construction() -> None:
    """Three ways to build a point cloud."""
    print("\n== construction ==")

    # From an (N, 3) array of any real dtype; always promoted to f8.
    xyz = np.random.default_rng(0).normal(size=(1000, 3)) * 10.0
    pc = PointCloud.from_xyz(xyz)
    print(f"from_xyz: {len(pc)} points, xyz dtype = {pc.xyz.dtype}")

    # From a dict of dimension arrays.
    pc2 = PointCloud.from_numpy(
        {
            "xyz": xyz,
            "intensity": np.random.default_rng(1).integers(0, 65535, size=1000),
        }
    )
    print(f"from_numpy: intensity dtype = {pc2.dtypes['intensity']}")

    # From a file (see example_file_io below for a full round trip).
    print(f"PointCloud() empty: {len(PointCloud())} points")


def example_dimension_access() -> None:
    """Dict-style and attribute-style dimension access, with the dtype promise."""
    print("\n== dimension access ==")

    xyz = np.random.default_rng(2).normal(size=(500, 3))
    pc = PointCloud.from_xyz(xyz)

    # Standard dimensions carry pinned dtypes (ADR-0002): setters coerce
    # same-kind arrays, and reject or overflow-check anything else.
    pc["classification"] = np.zeros(len(pc), dtype=np.uint8)
    pc.intensity = np.arange(len(pc)) % 1000  # int64 list coerces to u2

    print(f"classification dtype: {pc.dtypes['classification']}")
    print(f"intensity dtype: {pc.dtypes['intensity']} (coerced from int64)")

    try:
        pc.classification = np.full(len(pc), 1.5)  # float into u1: rejected
    except TypeError as exc:
        print(f"float -> u1 correctly rejected: {exc}")

    try:
        pc.intensity = np.full(len(pc), -1, dtype=np.int32)  # out of range
    except OverflowError as exc:
        print(f"out-of-range value correctly rejected: {exc}")

    # Extra dimensions must be declared before use.
    pc.add_extra_dim("confidence", "f4")
    pc.confidence = np.random.default_rng(3).random(len(pc)).astype(np.float32)
    print(f"declared dims: {pc.dims}")


def example_geometry() -> None:
    """Coordinate transforms."""
    print("\n== geometry ==")

    xyz = np.random.default_rng(4).normal(size=(200, 3))
    pc = PointCloud.from_xyz(xyz)

    theta = math.radians(30)
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([1.0, 2.0, 3.0])

    pc_rigid = pc.rigid_transform(rotation, translation)
    print(f"rigid_transform: first point moved by {pc_rigid.xyz[0] - pc.xyz[0]}")

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    pc_transformed = pc.transform(matrix)
    np.testing.assert_allclose(pc_transformed.xyz, pc_rigid.xyz, atol=1e-3)
    print("transform(4x4) matches rigid_transform for the same rotation+translation")


def example_downsample() -> None:
    """Voxel downsampling strategies."""
    print("\n== voxel downsample ==")

    xyz = np.random.default_rng(5).normal(size=(50_000, 3)) * 10.0
    pc = PointCloud.from_xyz(xyz)

    pc_centroid = pc.voxel_downsample(
        0.5, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
    )
    pc_random = pc.voxel_downsample(0.5, strategy=DownsampleStrategy.RANDOM, seed=42)

    print(f"original: {len(pc):,} points")
    print(f"nearest-to-centroid: {len(pc_centroid):,} points")
    print(f"random (seed=42): {len(pc_random):,} points")


def example_device_selection() -> None:
    """Runtime device selection (ADR-0001)."""
    print("\n== device selection ==")

    print(f"available devices: {available_devices()}")
    print(f"default device: {default_device()}")

    pc = PointCloud.from_xyz(np.zeros((10, 3)))
    print(f"new cloud runs on: {pc.device}")


def example_file_io() -> None:
    """Read/write round trip across formats (ADR-0004)."""
    print("\n== file I/O ==")

    xyz = np.random.default_rng(6).normal(size=(1000, 3)) * 50.0
    pc = PointCloud.from_xyz(xyz)
    pc["intensity"] = np.random.default_rng(7).integers(0, 65535, size=len(pc))
    pc["classification"] = np.random.default_rng(8).integers(0, 10, size=len(pc))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cloud.parquet"
        pc.write(str(path))
        loaded = PointCloud.read(str(path))
        print(f"parquet round trip: {len(loaded)} points, dims={loaded.dims}")

        las_path = Path(tmpdir) / "cloud.laz"
        pc.write(str(las_path))
        loaded_laz = PointCloud.read(str(las_path))
        print(f"laz round trip: {len(loaded_laz)} points (compressed)")


def example_torch_interop() -> None:
    """Optional torch interop (ADR-0001) -- only runs if torch is installed."""
    print("\n== torch interop (optional) ==")

    try:
        import torch
    except ImportError:
        print("torch is not installed; skipping this example")
        return

    xyz = torch.rand((100, 3), dtype=torch.float64) * 10.0
    pc = PointCloud.from_xyz(xyz)  # zero-copy on CPU via DLPack
    print(f"built from a torch tensor: {len(pc)} points")

    tensor_x = pc.to_torch("x")
    print(f"to_torch('x'): {type(tensor_x).__name__}, dtype={tensor_x.dtype}")


if __name__ == "__main__":
    example_construction()
    example_dimension_access()
    example_geometry()
    example_downsample()
    example_device_selection()
    example_file_io()
    example_torch_interop()
