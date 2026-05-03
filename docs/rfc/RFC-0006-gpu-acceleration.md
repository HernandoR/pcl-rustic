# RFC-0006: GPU Acceleration Backend

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-05-03 |
| **Updated** | 2026-05-03 |
| **Related RFCs** | [RFC-0001](RFC-0001-core-architecture.md) · [RFC-0003](RFC-0003-advanced-downsampling.md) · [RFC-0004](RFC-0004-point-cloud-registration.md) · [RFC-0005](RFC-0005-normal-estimation.md) · [RFC-0007](RFC-0007-testing-infrastructure.md) |

## Summary

Enable GPU acceleration for compute-intensive operations (voxel downsampling, FPS, normal estimation, ICP distance computations) via **Burn's backend abstraction**, exposing `wgpu` (cross-platform, no CUDA driver required) as the default GPU path and `tch` (LibTorch/CUDA) as an optional high-performance backend. Backend selection is automatic with a manual override via an environment variable and Python API.

## Motivation

The NdArray backend used in RFC-0001 executes all tensor operations on CPU. Profiling shows that for clouds > 5 M points:

- Voxel downsampling: 7 s on M1 CPU → target < 1 s on GPU.
- FPS (RFC-0003): O(N·k) is prohibitively slow on CPU for large N.
- Normal estimation PCA (RFC-0005): parallelises perfectly on GPU.
- ICP distance matrix (RFC-0004): N×M distances → ideal GPU workload.

Burn's backend trait allows swapping the compute engine without changing algorithm code, making GPU support low-risk to introduce.

## Design

### 6.1 Backend Hierarchy

```
Burn Backend Trait
├── NdArrayBackend     (CPU, always available, zero extra deps)
├── WgpuBackend        (GPU via WebGPU/Vulkan/Metal/DX12, no CUDA driver needed)
└── TchBackend         (CUDA/ROCm via LibTorch, optional feature flag)
```

Wgpu is the default GPU backend because it:
- Works on Linux (Vulkan), macOS (Metal), Windows (DX12/Vulkan), and web (WebGPU).
- Requires no proprietary NVIDIA driver.
- Ships as a pure-Rust crate (no C++ dependencies).

### 6.2 Cargo Feature Flags

```toml
[features]
default  = ["cpu"]
cpu      = ["burn/ndarray"]
gpu-wgpu = ["burn/wgpu"]
gpu-cuda = ["burn/tch", "tch/cuda-bundle"]
```

Users install the appropriate variant:

```bash
# CPU-only (default)
pip install pcl-rustic

# GPU via wgpu (cross-platform)
pip install pcl-rustic[wgpu]

# CUDA (Linux/Windows, NVIDIA)
pip install pcl-rustic[cuda]
```

### 6.3 Backend Selection

**Automatic**: at module import, PCL Rustic probes for available hardware:

```python
# Auto-selected at import time; can be overridden
import pcl_rustic
print(pcl_rustic.backend())   # "wgpu" | "ndarray" | "tch-cuda"
```

**Manual override**:

```python
import os
os.environ["PCL_RUSTIC_BACKEND"] = "ndarray"   # force CPU
import pcl_rustic
```

**Runtime switch** (for testing):

```python
pcl_rustic.set_backend("wgpu")     # raises if backend not compiled in
pcl_rustic.set_backend("ndarray")  # always works
```

### 6.4 Burn Tensor Integration

Replace the current `Vec<Vec<f32>>` storage in `HighPerformancePointCloud` with a Burn `Tensor<B, 2>` to enable zero-copy GPU operations:

```rust
pub struct HighPerformancePointCloud<B: Backend> {
    xyz:        Tensor<B, 2>,                      // shape [M, 3]
    intensity:  Option<Tensor<B, 1>>,              // shape [M]
    rgb:        Option<Tensor<B, 2>>,              // shape [M, 3]
    attributes: HashMap<String, Tensor<B, 1>>,
    _backend:   PhantomData<B>,
}
```

> **Migration note**: this is a breaking change to the Rust internal API. Python-facing API remains identical thanks to the PyO3 wrapper.

The Python-facing `PyPointCloud` wraps an `enum BackendCloud { Ndarray(...), Wgpu(...), Tch(...) }` to avoid monomorphisation leaking into Python bindings.

### 6.5 GPU-Accelerated Operations

| Operation | GPU implementation |
|-----------|-------------------|
| Voxel hash (RFC-0001) | Parallel hash-bucket assignment on GPU |
| FPS (RFC-0003) | Parallel distance update with `argmax` reduction |
| Normal PCA (RFC-0005) | Batched 3×3 SVD on neighbour matrices |
| ICP distance matrix (RFC-0004) | Batched nearest-neighbour with pairwise `L2` |

### 6.6 NumPy Interop on GPU

When returning data to Python, tensors are pulled back to CPU before NumPy conversion. The transfer is explicit and synchronous to avoid race conditions:

```rust
fn get_xyz(&self, py: Python<'_>) -> PyResult<PyObject> {
    let cpu_tensor = self.xyz.clone().into_data().to_vec::<f32>()?;
    // ... convert to numpy
}
```

For advanced users, an async `to_numpy_async()` method may be added in a future RFC.

### Alternatives Considered

| Approach | Reason Not Chosen |
|----------|-------------------|
| Direct CUDA kernels (cudarc) | Linux/NVIDIA only; no portability |
| OpenCL | Deprecated on macOS, poor Rust ecosystem |
| CUDAExecutionProvider (ONNX Runtime) | Adds onnxruntime dependency; overkill |
| Python CuPy interop | Python-side, no zero-copy path |

## Impact

### Breaking Changes

- `HighPerformancePointCloud` becomes generic over `B: Backend` in Rust. **Python API is unchanged** — the generic is erased at the PyO3 boundary.
- Wheel packages split into `cpu`, `gpu-wgpu`, `gpu-cuda` extras.

### Performance Implications

Expected speedups (GPU vs CPU, wgpu backend, NVIDIA RTX 3080):

| Operation | CPU (NdArray) | GPU (wgpu) | Speedup |
|-----------|---------------|------------|---------|
| Voxel 10M pts | 7 s | ~0.8 s | ~9× |
| FPS 1M → 1k | 5 s | ~0.3 s | ~17× |
| Normal est. 1M | 1 s | ~0.1 s | ~10× |

### Testing Plan

- `tests/test_gpu.py`:
  - Tests are **skipped** (pytest `skipIf`) when no GPU is available.
  - CPU and GPU backends produce numerically identical results within `atol=1e-4`.
  - Backend selection via env var is respected.
  - Memory stays on GPU between consecutive operations (no spurious CPU roundtrips).
- CI:
  - Default CI (GitHub Actions free tier) runs CPU-only tests.
  - GPU CI jobs (self-hosted runner or cloud GPU) run `tests/test_gpu.py`; optional/non-blocking.
  - See [RFC-0007](RFC-0007-testing-infrastructure.md) for CI matrix details.
- API docs: use context7 to verify Burn `wgpu` feature flags and tensor API.

## Unresolved Questions

1. Should `wgpu` become the default for PyPI wheels, adding ~50 MB to wheel size?
2. How do we handle GPU out-of-memory (OOM) errors gracefully — fallback to CPU or propagate?
3. Should Burn's `Autodiff<B>` wrapper be exposed for gradient-based optimisation (future ML RFC)?
4. Wheel naming conventions for GPU variants — PEP 425 does not have a GPU tag.

## References

- [Burn Framework — Backends](https://burn.dev/docs/concepts/backend)
- [wgpu — cross-platform GPU](https://wgpu.rs/)
- [tch-rs — LibTorch bindings](https://github.com/LaurentMazare/tch-rs)
- [RFC-0001: Core Architecture](RFC-0001-core-architecture.md)
- [RFC-0003: Advanced Downsampling](RFC-0003-advanced-downsampling.md)
- [RFC-0004: Point Cloud Registration](RFC-0004-point-cloud-registration.md)
- [RFC-0005: Normal Vector Estimation](RFC-0005-normal-estimation.md)
- [RFC-0007: Testing & Developer Infrastructure](RFC-0007-testing-infrastructure.md)
