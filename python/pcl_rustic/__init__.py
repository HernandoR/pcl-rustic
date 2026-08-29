"""pcl-rustic: a Rust + PyO3 + burn point cloud library.

This module wraps the compiled `._core` extension by composition. Every
`PointCloud` holds a `._core.PointCloud` instance (`_inner`) and forwards
work to it; the wrapper's job is the numpy/torch boundary (ADR-0001) and the
laspy-aligned dtype contract (ADR-0002), not point cloud logic itself.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from . import _core
from ._core import DownsampleStrategy, available_devices, default_device
from ._core import device_report as _device_report

__version__ = "0.1.0"

__all__ = [
    "PointCloud",
    "DownsampleStrategy",
    "read",
    "available_devices",
    "default_device",
    "device_report",
    "STANDARD_DIMENSIONS",
]


def device_report() -> dict[str, str]:
    """Why each device is or is not usable on this machine.

    Maps every device name this build knows about to ``"available"``, ``"not
    compiled into this build"``, or ``"unavailable: <reason>"``. Unlike
    :func:`available_devices`, which only reports *that* a GPU is missing,
    this reports *why* -- so a silent fallback to ``cpu`` on a host that does
    have a GPU is diagnosable.

    On Linux the usual cause is a missing Vulkan **loader**: NVIDIA's driver
    installs an ICD at ``/etc/vulkan/icd.d/`` but not ``libvulkan.so.1``, so
    wgpu enumerates zero adapters. Install the loader (``apt install
    libvulkan1``) and ``vulkan`` appears.

    Probing a GPU is not free, and the result is not cached here, so call
    this when diagnosing rather than on a hot path.
    """
    return dict(_device_report())


#: Standard dimension names and their pinned numpy dtypes, per ADR-0002.
STANDARD_DIMENSIONS: dict[str, np.dtype] = {
    "x": np.dtype(np.float64),
    "y": np.dtype(np.float64),
    "z": np.dtype(np.float64),
    "intensity": np.dtype(np.uint16),
    "return_number": np.dtype(np.uint8),
    "number_of_returns": np.dtype(np.uint8),
    "synthetic": np.dtype(np.bool_),
    "key_point": np.dtype(np.bool_),
    "withheld": np.dtype(np.bool_),
    "overlap": np.dtype(np.bool_),
    "scan_direction_flag": np.dtype(np.bool_),
    "edge_of_flight_line": np.dtype(np.bool_),
    "scanner_channel": np.dtype(np.uint8),
    "classification": np.dtype(np.uint8),
    "user_data": np.dtype(np.uint8),
    "scan_angle": np.dtype(np.int16),
    "point_source_id": np.dtype(np.uint16),
    "gps_time": np.dtype(np.float64),
    "red": np.dtype(np.uint16),
    "green": np.dtype(np.uint16),
    "blue": np.dtype(np.uint16),
    "nir": np.dtype(np.uint16),
}

#: Allowed dtypes for `add_extra_dim`, per ADR-0002 (laspy's ExtraBytes set).
_ALLOWED_EXTRA_DTYPES: dict[str, str] = {
    "uint8": "u1",
    "uint16": "u2",
    "uint32": "u4",
    "uint64": "u8",
    "int8": "i1",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
    "float32": "f4",
    "float64": "f8",
}


def _torch_module() -> Any | None:
    """Return the `torch` module if importable, else None. Never raises."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _require_torch() -> Any:
    torch = _torch_module()
    if torch is None:
        raise ImportError(
            "torch is required for this operation; install it with "
            "`pip install torch` or `uv add torch`"
        )
    return torch


def _to_ndarray(value: ArrayLike) -> np.ndarray:
    """Convert numpy/torch/list/buffer input to a plain numpy array.

    Torch tensors are ingested zero-copy on CPU via DLPack; off-device
    tensors are copied to host first (ADR-0001 §Torch-tensor compatibility).
    """
    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return np.from_dlpack(tensor)
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _cast_same_kind(arr: np.ndarray, target: np.dtype) -> np.ndarray:
    """Coerce `arr` to `target` using laspy-style setter semantics.

    Same-kind numeric casts (int<->int, float<->float) are allowed subject
    to range validation (`OverflowError` on out-of-range values);
    kind-incompatible casts (e.g. float into an integer dimension) raise
    `TypeError`. This mirrors laspy's point-record assignment behavior
    (ADR-0002).
    """
    src_kind = arr.dtype.kind
    tgt_kind = target.kind

    if tgt_kind == "b":
        if src_kind != "b":
            raise TypeError(
                f"cannot assign an array of dtype {arr.dtype} to a bool dimension"
            )
        return arr.astype(target)

    if tgt_kind in ("i", "u"):
        if src_kind not in ("i", "u"):
            raise TypeError(
                f"cannot assign an array of dtype {arr.dtype} to an integer "
                f"dimension of dtype {target}"
            )
        info = np.iinfo(target)
        if arr.size and (int(arr.min()) < info.min or int(arr.max()) > info.max):
            raise OverflowError(f"value out of range for dtype {target}")
        return arr.astype(target)

    if tgt_kind == "f":
        if src_kind not in ("i", "u", "f"):
            raise TypeError(
                f"cannot assign an array of dtype {arr.dtype} to a float "
                f"dimension of dtype {target}"
            )
        casted = arr.astype(target)
        if target.itemsize < arr.dtype.itemsize:
            finite_before = np.isfinite(arr)
            finite_after = np.isfinite(casted)
            if np.any(finite_before & ~finite_after):
                raise OverflowError(f"value out of range for dtype {target}")
        return casted

    raise TypeError(f"cannot assign an array of dtype {arr.dtype} to dtype {target}")


def _coerce(value: ArrayLike, target_dtype: DTypeLike) -> np.ndarray:
    """Convert `value` (numpy/torch/list) to an array of `target_dtype`.

    An array already at the exact dtype passes through unchanged; anything
    else goes through `_cast_same_kind`.
    """
    target = np.dtype(target_dtype)
    arr = _to_ndarray(value)
    if arr.dtype == target:
        return np.ascontiguousarray(arr)
    return _cast_same_kind(arr, target)


class PointCloud:
    """A set of named, equal-length point dimensions (ADR-0002).

    Wraps a `._core.PointCloud` by composition. Dimensions are accessed
    uniformly via `pc["name"]` / `pc["name"] = array`, with attribute sugar
    (`pc.intensity`, `pc.classification`, ...) for standard and already
    declared extra dimensions. `x`, `y`, `z`, and `xyz` are always
    available as properties.
    """

    _inner: _core.PointCloud

    def __init__(self) -> None:
        object.__setattr__(self, "_inner", _core.PointCloud())

    @classmethod
    def _wrap(cls, inner: _core.PointCloud) -> "PointCloud":
        """Wrap an existing `_core.PointCloud`, bypassing `__init__`."""
        obj = cls.__new__(cls)
        object.__setattr__(obj, "_inner", inner)
        return obj

    # -- construction ------------------------------------------------

    @classmethod
    def from_xyz(cls, xyz: ArrayLike) -> "PointCloud":
        """Build a point cloud from an (N, 3) array of any real dtype.

        The input is coerced to float64 before reaching `_core`, so the
        `x`/`y`/`z` dtype promise (ADR-0002) holds regardless of input
        dtype.
        """
        arr = _coerce(xyz, np.float64)
        return cls._wrap(_core.PointCloud.from_xyz(arr))

    @classmethod
    def from_numpy(cls, data: Mapping[str, ArrayLike]) -> "PointCloud":
        """Build a point cloud from a dict of dimension name -> array.

        Requires either an `"xyz"` key or all of `"x"`, `"y"`, `"z"`. Every
        other key becomes a dimension: standard names use their pinned
        dtype; unknown names become extra dimensions, inferring the
        `add_extra_dim` dtype from the array's own dtype (must be one of
        the allowed extra-dimension dtypes).
        """
        keys = set(data.keys())
        if "xyz" in keys:
            xyz = data["xyz"]
        elif {"x", "y", "z"} <= keys:
            x = _coerce(data["x"], np.float64)
            y = _coerce(data["y"], np.float64)
            z = _coerce(data["z"], np.float64)
            xyz = np.stack([x, y, z], axis=1)
        else:
            raise ValueError(
                "PointCloud.from_numpy requires an 'xyz' key or all of 'x', 'y', 'z'"
            )

        pc = cls.from_xyz(xyz)
        used = {"xyz", "x", "y", "z"} & keys
        for name, value in data.items():
            if name in used:
                continue
            pc._assign_or_declare_dim(name, value)
        return pc

    @classmethod
    def read(
        cls,
        path: str,
        columns: dict[str, str] | None = None,
        delimiter: str = ",",
    ) -> "PointCloud":
        """Read a point cloud from `.las`/`.laz`/`.csv`/`.parquet` (ADR-0004)."""
        return cls._wrap(_core.PointCloud.read(path, columns, delimiter))

    # -- dimension access ---------------------------------------------

    def _assign_dim(self, name: str, value: ArrayLike) -> None:
        """Coerce `value` to the right dtype for `name` and store it.

        Standard names use their pinned dtype (ADR-0002 table). Existing
        extra dimensions use their declared dtype. Unknown names that are
        not yet declared fall through to `_core.set_dim`, which raises
        `KeyError` (extra dimensions must be declared via `add_extra_dim`
        first).
        """
        if name == "xyz" or name in ("x", "y", "z"):
            arr = _coerce(value, np.float64)
        elif name in STANDARD_DIMENSIONS:
            arr = _coerce(value, STANDARD_DIMENSIONS[name])
        else:
            dtypes = self._inner.dim_dtypes()
            if name in dtypes:
                arr = _coerce(value, np.dtype(dtypes[name]))
            else:
                arr = _to_ndarray(value)
        self._inner.set_dim(name, arr)

    def _assign_or_declare_dim(self, name: str, value: ArrayLike) -> None:
        """Like `_assign_dim`, but auto-declares an unknown name as an
        extra dimension first, inferring its dtype from `value`'s own
        dtype. Used by `from_numpy`; direct `pc[name] = value` assignment
        still requires `add_extra_dim` to be called explicitly first.
        """
        if name in STANDARD_DIMENSIONS or self._inner.has_dim(name):
            self._assign_dim(name, value)
            return
        arr = _to_ndarray(value)
        letter = _ALLOWED_EXTRA_DTYPES.get(arr.dtype.name)
        if letter is None:
            raise TypeError(
                f"cannot infer an extra-dimension dtype for '{name}' from "
                f"array dtype {arr.dtype}; declare it explicitly with "
                "add_extra_dim()"
            )
        self._inner.add_extra_dim(name, letter)
        self._inner.set_dim(name, arr)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._inner.get_dim(name)

    def __setitem__(self, name: str, value: ArrayLike) -> None:
        self._assign_dim(name, value)

    def __getattr__(self, name: str) -> np.ndarray:
        # Only called when normal attribute lookup (instance dict, class
        # attributes, properties) fails to find `name`.
        if name.startswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        if name in STANDARD_DIMENSIONS or inner.has_dim(name):
            try:
                return inner.get_dim(name)
            except KeyError as exc:
                raise AttributeError(name) from exc
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_inner":
            object.__setattr__(self, name, value)
            return
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
            return
        inner = self.__dict__.get("_inner")
        if inner is not None and (name in STANDARD_DIMENSIONS or inner.has_dim(name)):
            self._assign_dim(name, value)
            return
        object.__setattr__(self, name, value)

    def has_dim(self, name: str) -> bool:
        return self._inner.has_dim(name)

    def remove_dim(self, name: str) -> None:
        self._inner.remove_dim(name)

    def add_extra_dim(self, name: str, dtype: str) -> None:
        """Declare a zero-filled extra dimension.

        `dtype` accepts either laspy-style letter codes (`u1`, `i2`, `f4`,
        ...) or numpy dtype names (`uint8`, `int16`, `float32`, ...).
        """
        self._inner.add_extra_dim(name, dtype)

    @property
    def dims(self) -> list[str]:
        return self._inner.dim_names()

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return {name: np.dtype(dt) for name, dt in self._inner.dim_dtypes().items()}

    # -- x/y/z/xyz sugar ------------------------------------------------

    @property
    def x(self) -> NDArray[np.float64]:
        return self._inner.get_dim("x")

    @x.setter
    def x(self, value: ArrayLike) -> None:
        self._assign_dim("x", value)

    @property
    def y(self) -> NDArray[np.float64]:
        return self._inner.get_dim("y")

    @y.setter
    def y(self, value: ArrayLike) -> None:
        self._assign_dim("y", value)

    @property
    def z(self) -> NDArray[np.float64]:
        return self._inner.get_dim("z")

    @z.setter
    def z(self, value: ArrayLike) -> None:
        self._assign_dim("z", value)

    @property
    def xyz(self) -> NDArray[np.float64]:
        return self._inner.get_dim("xyz")

    @xyz.setter
    def xyz(self, value: ArrayLike) -> None:
        self._assign_dim("xyz", value)

    # -- torch interop --------------------------------------------------

    def to_torch(self, name: str | None = None) -> Any:
        """Export one dimension, or every dimension, as torch tensor(s).

        Zero-copy via `torch.from_numpy` (the source array is a fresh copy
        already, per ADR-0002's getter semantics). Raises `ImportError`
        with an actionable message if torch is not installed.
        """
        torch = _require_torch()
        if name is not None:
            return torch.from_numpy(self[name])
        return {dim: torch.from_numpy(self[dim]) for dim in self.dims}

    # -- geometry ---------------------------------------------------------

    def transform(self, matrix: ArrayLike) -> "PointCloud":
        arr = _coerce(matrix, np.float64)
        return self._wrap(self._inner.transform(arr))

    def rigid_transform(
        self, rotation: ArrayLike, translation: ArrayLike
    ) -> "PointCloud":
        rot = _coerce(rotation, np.float64)
        trans = _coerce(translation, np.float64)
        return self._wrap(self._inner.rigid_transform(rot, trans))

    def voxel_downsample(
        self,
        voxel_size: float,
        strategy: int = DownsampleStrategy.NEAREST_TO_CENTROID,
        seed: int | None = None,
    ) -> "PointCloud":
        return self._wrap(self._inner.voxel_downsample(voxel_size, strategy, seed))

    # -- device -------------------------------------------------------------

    @property
    def device(self) -> str:
        return self._inner.device

    def to_device(self, device: str) -> "PointCloud":
        return self._wrap(self._inner.to_device(device))

    # -- I/O, misc ------------------------------------------------------

    def write(
        self,
        path: str,
        columns: dict[str, str] | None = None,
        delimiter: str = ",",
    ) -> None:
        self._inner.write(path, columns, delimiter)

    def clone(self) -> "PointCloud":
        return self._wrap(self._inner.clone())

    def __len__(self) -> int:
        return len(self._inner)

    def __repr__(self) -> str:
        return repr(self._inner)


def read(
    path: str,
    columns: dict[str, str] | None = None,
    delimiter: str = ",",
) -> PointCloud:
    """Module-level alias mirroring `laspy.read`. See `PointCloud.read`."""
    return PointCloud.read(path, columns=columns, delimiter=delimiter)
