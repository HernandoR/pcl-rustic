# RFC-0002: Tensor Backend & GPU Acceleration

- **Status**: Draft
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-05-03
- **Updated**: 2026-05-03

---

## Summary

Introduce a configurable Burn tensor backend selection mechanism that allows pcl-rustic to execute heavy batch operations on GPU (WGPU/CUDA) when available, falling back transparently to the CPU NdArray backend.  
This RFC defines the backend selection API, Python-side configuration, and the migration path for all compute-heavy kernels.

---

## Motivation

The current implementation always uses Burn's `NdArray` CPU backend.  
For point clouds exceeding ~10M points the voxel-grid, registration, and segmentation algorithms become the runtime bottleneck.  
WGPU (WebGPU-based cross-platform GPU compute) and CUDA backends are already available in Burn `≥0.20`, making GPU acceleration feasible without changing algorithm code.

Pain points:
- Voxel downsample of 50M points takes ~35–47 s on CPU (see README benchmark table).
- ICP and NDT (RFC-0005) will be order-of-magnitude slower on CPU than GPU.
- Users cannot choose the backend without recompiling.

---

## Detailed Design

### Backend Variants

```rust
pub enum BackendKind {
    NdArray,   // CPU default, always available
    Wgpu,      // WebGPU, cross-platform GPU
    Cuda,      // NVIDIA CUDA (feature-flag, optional)
    // future: Metal, Vulkan
}
```

### Runtime Selection

Burn backends are compile-time generics, but we can select between pre-compiled backend implementations using Rust enums + dynamic dispatch via a `dyn ComputeBackend` trait.

```rust
pub trait ComputeBackend: Send + Sync {
    fn voxel_bin(&self, xyz: &[f32], voxel_size: f32) -> Vec<u64>;
    fn nearest_in_voxel(&self, xyz: &[f32], bins: &[u64], strategy: u8) -> Vec<usize>;
    // … other heavy kernels
}
```

Two concrete implementations will be compiled into the `.so`:
- `NdArrayBackend` — always available
- `WgpuBackend` — enabled when `burn/wgpu` feature is active (default)

### Python API

```python
from pcl_rustic import set_backend, get_backend, BackendKind

# Select before any computation
set_backend(BackendKind.WGPU)    # GPU (default if available)
set_backend(BackendKind.NDARRAY) # force CPU

# Read current
backend = get_backend()
print(backend)  # "wgpu" or "ndarray"
```

`set_backend` stores the selection in a thread-local (later: global config object).  
All heavy operations consult this selection at call time.

### `PointCloud` constructor enhancement

```python
pc = PointCloud.from_xyz(xyz, backend=BackendKind.WGPU)
```

The backend is stored on the `PointCloud` instance and inherited by derived clouds (transform, downsample, etc.).

### Feature Flags (Cargo.toml)

```toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]
cuda = ["burn/cuda"]
ndarray-only = []   # CI/minimal builds
```

### Graceful Fallback

If WGPU device initialisation fails at runtime (no GPU, driver issue), the library logs a warning and falls back to NdArray automatically:

```python
import warnings
warnings.warn("WGPU backend unavailable, falling back to NdArray", RuntimeWarning)
```

---

## Implementation Plan

- [ ] Add `BackendKind` enum to `src/lib.rs` PyO3 bindings
- [ ] Implement `ComputeBackend` trait in `src/backends/`
  - [ ] `src/backends/ndarray.rs`
  - [ ] `src/backends/wgpu.rs`
- [ ] Refactor `src/point_cloud/voxel.rs` to call `ComputeBackend` trait
- [ ] Add `set_backend` / `get_backend` Python API
- [ ] Add Cargo feature flags `wgpu` (default) and `cuda` (optional)
- [ ] Update `justfile` with `just build-cuda` target
- [ ] Add pytest tests: `tests/test_backend.py`
- [ ] Update CI matrix to include a WGPU smoke test (headless)
- [ ] Update docs: `docs/performance/optimization.md`

---

## Testing Strategy

```python
# tests/test_backend.py
import pytest
from pcl_rustic import BackendKind, PointCloud, set_backend, get_backend
import numpy as np

def test_default_backend():
    backend = get_backend()
    assert backend in (BackendKind.WGPU, BackendKind.NDARRAY)

def test_set_ndarray_backend():
    set_backend(BackendKind.NDARRAY)
    assert get_backend() == BackendKind.NDARRAY

def test_backend_inherited_by_clone():
    set_backend(BackendKind.NDARRAY)
    xyz = np.random.randn(100, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc2 = pc.clone()
    assert pc2.backend == BackendKind.NDARRAY

@pytest.mark.slow
def test_wgpu_backend_large_cloud():
    set_backend(BackendKind.WGPU)
    xyz = np.random.randn(10_000_000, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    result = pc.voxel_downsample(0.15)
    assert result.point_count() > 0
```

Run with `just test` for fast subset, `just test-slow` for GPU tests.

---

## Alternatives Considered

- **CuPy/Torch FFI**: adds large Python runtime dependencies; not zero-copy with Rust.
- **Separate `pcl-rustic-gpu` package**: worse DX; users would have to install two packages.
- **Compile-time static backend**: current state; prevents runtime flexibility.

---

## Drawbacks

- Binary size increases (WGPU shader compilation adds ~5–10 MB).
- WGPU requires a GPU driver or software renderer (lavapipe/swiftshader) in CI.
- `ComputeBackend` dynamic dispatch adds a pointer indirection per kernel call (negligible overhead).

---

## Unresolved Questions

- Should `BackendKind.CUDA` be a separate optional wheel on PyPI to avoid CUDA driver dependency for non-GPU users?
- What is the minimum Vulkan/Metal driver version required for WGPU?
- Should we expose a `pcl_rustic.device_info()` API to report available hardware?

---

## Related RFCs

- [RFC-0001](./RFC-0001-foundation.md) — Foundational architecture; this RFC extends it with backend layer
- [RFC-0004](./RFC-0004-sampling-strategies.md) — FPS sampling benefits most from GPU kernels
- [RFC-0005](./RFC-0005-registration.md) — ICP/NDT inner loops are primary GPU acceleration targets
- [RFC-0006](./RFC-0006-normal-estimation.md) — KNN search for normals is GPU-acceleratable
- [RFC-0007](./RFC-0007-segmentation.md) — DBSCAN/region-growing benefit from GPU-parallel distance computation
