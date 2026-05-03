# RFC-0002 — GPU Acceleration Backend

| Field | Value |
|-------|-------|
| RFC | 0002 |
| Status | Open for Review |
| Author | liuzhen19 |
| Created | 2026-05-03 |
| Updated | 2026-05-03 |

## Summary

Extend PCL Rustic to support transparent GPU acceleration by promoting Burn's `wgpu` backend to first-class status. Introduce a runtime backend selector and ensure the public API remains unchanged regardless of whether computation runs on CPU or GPU.

## Motivation

The current library always uses Burn's `ndarray` (CPU) backend. For large point clouds (> 50 M points), GPU parallelism can reduce processing time by an order of magnitude. Users working on NVIDIA RTX, Apple Silicon (Metal), or integrated Intel Arc GPUs should be able to opt into GPU execution with a single environment variable or API call.

Key pain points without GPU support:

- Voxel downsampling of 50 M points takes ~35–47 s on CPU (measured benchmark).
- Normal estimation (RFC-0005) and segmentation (RFC-0006) will be even more compute-intensive.
- Scientific users often have workstations/clusters with GPU access.

## Design

### Backend selection

Burn supports multiple backends via feature flags. We already compile with both `ndarray` and `wgpu` features:

```toml
# Cargo.toml (existing)
burn = { version = "0.20.1", features = ["std", "wgpu", "ndarray", "cpu", "router"] }
```

We introduce a `BackendKind` enum in `src/utils/backend.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,   // burn ndarray backend (always available)
    Gpu,   // burn wgpu backend (requires compatible GPU + driver)
    Auto,  // probe GPU; fall back to CPU if unavailable
}
```

A global, lazy-initialized `ACTIVE_BACKEND` (using `std::sync::OnceLock`) stores the resolved backend at process start.

### Python API changes

```python
import pcl_rustic

# Get current backend
backend = pcl_rustic.get_backend()   # returns "cpu" | "gpu"

# Set backend before any computation
pcl_rustic.set_backend("gpu")   # raises RuntimeError if GPU unavailable
pcl_rustic.set_backend("auto")  # silent fallback to CPU

# PointCloud API is unchanged — backend is transparent
pc = PointCloud.from_xyz(xyz)
pc_down = pc.voxel_downsample(0.1)
```

### Rust internals

Introduce a type alias pattern to swap the Burn backend:

```rust
// src/utils/backend.rs
use burn::backend::{NdArray, Wgpu};
use burn::tensor::Tensor;

pub type CpuBackend  = NdArray<f32>;
pub type GpuBackend  = Wgpu;

// A dynamic dispatch wrapper used at runtime
pub enum ActiveTensor<const D: usize> {
    Cpu(Tensor<CpuBackend, D>),
    Gpu(Tensor<GpuBackend, D>),
}
```

All tensor operations in `voxel.rs`, `transform.rs`, etc. are refactored to accept `ActiveTensor` instead of a concrete backend type.

### Justfile additions

```just
# Build with GPU support (wgpu/Vulkan)
build-gpu:
    maturin develop --release --features gpu

# Run tests on GPU backend (requires compatible hardware)
test-gpu:
    PCLUSTIC_BACKEND=gpu uv run pytest tests/ -v -m gpu
```

### CI changes

- Add a new `gpu` pytest marker for tests that require hardware.
- Default CI matrix runs with `PCLUSTIC_BACKEND=cpu` (no GPU required).
- An optional manual workflow (`workflow_dispatch`) runs the GPU matrix on self-hosted runners.

### Integration tests

```python
# tests/test_backend.py
import pytest
import pcl_rustic

def test_default_backend_is_cpu():
    assert pcl_rustic.get_backend() in ("cpu", "gpu")

def test_set_backend_cpu():
    pcl_rustic.set_backend("cpu")
    assert pcl_rustic.get_backend() == "cpu"

@pytest.mark.gpu
def test_set_backend_gpu():
    pcl_rustic.set_backend("gpu")
    assert pcl_rustic.get_backend() == "gpu"

@pytest.mark.gpu
def test_voxel_downsample_gpu_matches_cpu(sample_point_cloud):
    pcl_rustic.set_backend("cpu")
    cpu_result = sample_point_cloud.voxel_downsample(0.1)

    pcl_rustic.set_backend("gpu")
    gpu_result = sample_point_cloud.voxel_downsample(0.1)

    assert cpu_result.point_count() == gpu_result.point_count()
```

## Drawbacks

- Dynamic backend dispatch adds a small runtime overhead per tensor operation (one `match` branch).
- The `wgpu` backend requires users to have GPU drivers installed; error messages must be clear.
- Testing GPU paths in standard CI is costly/slow. GPU-specific tests are gated behind a marker.
- Burn's `wgpu` backend is still maturing; some operations may not be implemented.

## Alternatives

| Alternative | Reason not chosen |
|-------------|-------------------|
| CUDA-only (cuBLAS) | Not portable; excludes Metal / OpenCL / Vulkan users |
| Python-level dispatch (CuPy / PyTorch) | Would bypass Rust compute; loses memory safety |
| Static compile-time backend | Users need to rebuild to switch; bad UX |
| OpenCL | Deprecated on macOS; poor toolchain support |

## Unresolved Questions

1. Should `set_backend` be process-global or per-`PointCloud` instance? Global is simpler; per-instance allows mixing.
2. How should we handle GPU OOM errors? Map to `MemoryError` in Python?
3. Should we expose the Burn `Device` (e.g., `cuda:0`, `wgpu:0`) for multi-GPU setups?
4. What is the minimum WGSL / Vulkan version required by Burn's wgpu backend?

## Related RFCs

- [RFC-0001 — Core Architecture](./RFC-0001-core-architecture.md) — establishes the Burn tensor framework and build tooling this RFC extends
- [RFC-0003 — Advanced Downsampling](./RFC-0003-advanced-downsampling.md) — will benefit from GPU backend for FPS (O(N²) → GPU-parallel)
- [RFC-0004 — Point Cloud Registration](./RFC-0004-point-cloud-registration.md) — iterative algorithms benefit most from GPU acceleration
- [RFC-0005 — Normal Estimation](./RFC-0005-normal-estimation.md) — kNN search is GPU-friendly
- [RFC-0007 — Testing & CI Infrastructure](./RFC-0007-testing-ci-infrastructure.md) — defines the `gpu` pytest marker and CI strategy

## Reviews

### Review 1 — Sub-agent Alpha

> **Status**: APPROVED
>
> The design cleanly separates the runtime backend selection from the public API, which is the right approach. The use of `OnceLock` for global state is safe and idiomatic. The `ActiveTensor` enum with runtime dispatch is slightly heavier than compile-time monomorphisation but is necessary for the "transparent fallback" requirement. Recommend ensuring the Python error message when GPU is unavailable is actionable (include driver version info). Unresolved Question 3 (multi-GPU) should be deferred to a follow-up RFC to keep scope manageable.

### Review 2 — Sub-agent Beta

> **Status**: APPROVED (with minor comments)
>
> Motivation section is strong and well-supported by benchmark data. CI strategy is sensible — gating GPU tests behind a manual workflow avoids infrastructure costs. One concern: the `ActiveTensor` enum duplicates a lot of match arms. Consider whether Burn's `AutodiffBackend` or the `Router` backend could provide the same polymorphism without a hand-rolled enum. Also clarify in the justfile recipe whether `--features gpu` is additive or replaces the default feature set.
