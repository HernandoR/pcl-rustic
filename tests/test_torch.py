"""Torch-tensor interop at the numpy/torch boundary (ADR-0001).

Skipped entirely when torch is not installed: torch is an optional import,
never a hard dependency of pcl-rustic.
"""

from __future__ import annotations

import numpy as np
import pytest
from pcl_rustic import PointCloud

torch = pytest.importorskip("torch")


class TestFromXyzTorchTensor:
    def test_from_xyz_accepts_torch_tensor(self) -> None:
        xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        pc = PointCloud.from_xyz(xyz)
        assert len(pc) == 2
        np.testing.assert_allclose(pc.xyz, xyz.numpy(), atol=1e-4)

    def test_from_xyz_detaches_gradient_tracked_tensor(self) -> None:
        xyz = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
        pc = PointCloud.from_xyz(xyz)
        assert len(pc) == 3


class TestSetItemTorchTensor:
    def test_setitem_accepts_torch_tensor_with_coercion(self) -> None:
        pc = PointCloud.from_xyz(np.zeros((5, 3)))
        intensity = torch.arange(5, dtype=torch.int32)
        pc["intensity"] = intensity
        assert pc["intensity"].dtype == np.uint16
        np.testing.assert_array_equal(
            pc["intensity"], intensity.numpy().astype(np.uint16)
        )


class TestToTorch:
    def test_to_torch_single_dim(self) -> None:
        pc = PointCloud.from_xyz(np.zeros((4, 3)))
        pc["intensity"] = np.array([1, 2, 3, 4], dtype=np.uint16)
        tensor = pc.to_torch("intensity")
        assert isinstance(tensor, torch.Tensor)
        np.testing.assert_array_equal(tensor.numpy(), pc["intensity"])

    def test_to_torch_all_dims(self) -> None:
        pc = PointCloud.from_xyz(np.zeros((3, 3)))
        pc["intensity"] = np.array([1, 2, 3], dtype=np.uint16)
        tensors = pc.to_torch()
        assert set(tensors) == set(pc.dims)
        for name, tensor in tensors.items():
            assert isinstance(tensor, torch.Tensor)
            np.testing.assert_array_equal(tensor.numpy(), pc[name])

    def test_to_torch_round_trip_preserves_values(self) -> None:
        pc = PointCloud.from_xyz(np.zeros((6, 3)))
        pc.add_extra_dim("weight", "f4")
        weight = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        pc["weight"] = weight
        tensor = pc.to_torch("weight")
        np.testing.assert_allclose(tensor.numpy(), weight)
