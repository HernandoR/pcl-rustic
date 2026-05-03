# RFC-0002: GPU Acceleration Support

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Draft                              |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2026-05-03                         |
| Updated | 2026-05-03                         |

## Summary

Extend PCL Rustic's compute backend to support GPU acceleration via the [Burn](https://github.com/tracel-ai/burn) framework's WGPU backend, and optionally CUDA/Metal when the target platform supports it. This enables large-scale point-cloud operations to benefit from massively parallel GPU hardware with no changes to the Python-facing API.

## Motivation

The current CPU-only path achieves ~1.1–1.5 M pts/s for voxel downsampling on an M1 MacBook. For production LiDAR pipelines processing 50 M+ points in real time (≥10 Hz), this is a bottleneck. GPU acceleration is projected to yield 5–20× speedup for embarrassingly parallel operations such as voxelization, nearest-neighbor searches, and rigid transforms.

Burn's backend abstraction makes this architecturally clean: the same tensor operations run on CPU (`ndarray` backend) or GPU (`wgpu`/`cuda` backend) by switching a single Rust generic at compile time or initialization time.

## Design Details

### Backend Selection Strategy

Burn uses a generic `Backend` type parameter throughout its `Tensor<B, D>` API. PCL Rustic currently hardcodes `NdArray` as the backend.

**Proposed approach**: introduce a Rust `enum BackendChoice` and select the backend at library initialization time, exposing a Python-level `set_backend(name: str)` function.

```rust
// src/backend.rs (new file)
pub enum BackendChoice {
    Cpu,   // NdArray (always available)
    Wgpu,  // cross-platform GPU via WebGPU
    Cuda,  // NVIDIA CUDA (optional feature flag)
}
```

On the Python side:

```python
import pcl_rustic
pcl_rustic.set_backend("wgpu")   # or "cpu", "cuda"
pcl_rustic.get_backend()         # returns current backend name
```

### Cargo Feature Flags

| Feature     | Default | Description                          |
|-------------|---------|--------------------------------------|
| `cpu`       | ✅ yes  | NdArray backend (always on)          |
| `wgpu`      | ✅ yes  | WGPU cross-platform GPU backend      |
| `cuda`      | ❌ no   | NVIDIA CUDA backend (opt-in)         |
| `metal`     | ❌ no   | Apple Metal (experimental, opt-in)   |

Default wheel build: `cpu + wgpu`. CUDA wheels built separately for `linux-x86_64` only.

### Tensor Operation Dispatch

All tensor operations (currently in `src/utils/tensor.rs`) must be refactored to be generic over the Burn backend `B: Backend`. Key operations:

| Operation              | Module               |
|------------------------|----------------------|
| Voxel grid creation    | `point_cloud/voxel.rs` |
| Point binning / scatter| `point_cloud/voxel.rs` |
| Centroid computation   | `point_cloud/voxel.rs` |
| Rigid transform (3×3 × N×3) | `traits/transform.rs` |

### Memory Transfer Minimization

GPU acceleration only helps when data stays on GPU across multiple operations. A `Pipeline` abstraction is proposed so that a sequence of operations (downsample → transform → compute normals) transfers data to GPU once and back once:

```python
result = (
    pcl_rustic.Pipeline(pc)
    .voxel_downsample(voxel_size=0.1)
    .rigid_transform(R, t)
    .compute_normals(k=20)
    .collect()  # transfers result back to CPU NumPy
)
```

This is intentionally left for a follow-up RFC (see Open Questions).

### Wheel Building Strategy

| Target platform        | Backend       | Wheel suffix          |
|------------------------|---------------|-----------------------|
| `manylinux-x86_64`    | cpu + wgpu    | `*-cp3xx-*.whl`       |
| `manylinux-x86_64`    | cpu + cuda    | `*-cu12x-*.whl`       |
| `macos-aarch64`        | cpu + wgpu    | `*-macosx-*.whl`      |
| `windows-x86_64`       | cpu + wgpu    | `*-win-*.whl`         |

CUDA wheels require `nvidia-cuda-toolkit` on the build runner. A separate GitHub Actions job with `runs-on: ubuntu-latest-gpu` handles this.

### Python-Facing Changes

- New: `pcl_rustic.set_backend(name)` / `pcl_rustic.get_backend() -> str`
- New: `pcl_rustic.list_backends() -> list[str]` (returns available compiled backends)
- No changes to `PointCloud` methods or constructor; backend is transparent.

### Error Handling

When a backend is requested but not compiled in, raise `RuntimeError`:

```python
pcl_rustic.set_backend("cuda")
# RuntimeError: CUDA backend not compiled. Install pcl-rustic-cuda wheel.
```

## Alternatives Considered

### Alternative 1: CuPy Integration

Use CuPy arrays directly from Python and pass GPU pointers to Rust via DLPack. Rejected because:
- Requires CuPy as a runtime dependency (large install).
- Users must manage GPU memory manually.
- Only supports CUDA, not WGPU/Metal.

### Alternative 2: Torch Backend (via `tch-rs`)

Use PyTorch's CUDA tensors via `tch-rs`. Rejected because:
- PyTorch is a very large dependency (~2 GB).
- `tch-rs` licensing and version management is complex.
- Burn already provides a cleaner abstraction.

### Alternative 3: Numba CUDA in Python

Offload only Python-level loops to Numba CUDA. Rejected because:
- Does not help Rust-level computation.
- Adds Python-only dependency with JIT compile latency.

## Open Questions

1. **Dynamic vs. static backend selection**: Should the backend be selected at import time (static, via env var) or at runtime (dynamic, via `set_backend`)? Runtime switching between backends requires care around active tensors.
2. **CUDA wheel distribution**: Should CUDA wheels be published to PyPI or only to GitHub Releases to avoid cluttering the PyPI package?
3. **Pipeline abstraction**: Should the pipeline API be defined in this RFC or a separate RFC? (Recommended: separate RFC-0008.)
4. **Free-threaded Python (3.14t)**: Burn's async execution may conflict with GIL-free Python; needs testing.

## Implementation Plan

- [ ] Add `wgpu` feature flag to `Cargo.toml` (already partially present)
- [ ] Create `src/backend.rs` with `BackendChoice` enum and initialization logic
- [ ] Refactor `src/utils/tensor.rs` to be generic over `B: Backend`
- [ ] Expose `set_backend` / `get_backend` / `list_backends` via PyO3
- [ ] Add GPU-aware CI job (WGPU software renderer for CI, real GPU on self-hosted runner)
- [ ] Add pytest tests: `tests/test_backend.py`
  - `test_set_backend_cpu()` — verify CPU backend works
  - `test_set_backend_wgpu()` — verify WGPU backend works (software renderer)
  - `test_backend_produces_same_result()` — CPU vs GPU output within tolerance
  - `test_invalid_backend_raises()` — verify `RuntimeError` for unknown backend
- [ ] Update `justfile`: add `just test-gpu` recipe
- [ ] Update CI workflow to add GPU software-renderer test matrix
- [ ] Update `rfcs/README.md` index

## Related RFCs

- **Background**: [RFC-0001](./RFC-0001-initial-architecture.md) — initial architecture and Burn tensor choice
- **Enables** → [RFC-0003](./RFC-0003-downsampling-strategies.md): GPU-accelerated FPS downsampling
- **Enables** → [RFC-0004](./RFC-0004-point-cloud-registration.md): GPU-accelerated ICP/NDT
- **Enables** → [RFC-0005](./RFC-0005-normal-estimation.md): GPU-accelerated normal estimation
- **Enables** → [RFC-0006](./RFC-0006-segmentation.md): GPU-accelerated segmentation
- **Related** → RFC-0008 (planned): Pipeline abstraction for zero-copy GPU chaining

## References

- [Burn Backend Documentation](https://burn.dev/docs/burn/backend/)
- [Burn WGPU Backend](https://github.com/tracel-ai/burn/tree/main/crates/burn-wgpu)
- [Burn CUDA Backend](https://github.com/tracel-ai/burn/tree/main/crates/burn-cuda)
- [WebGPU specification](https://www.w3.org/TR/webgpu/)
- [RFC-0001](./RFC-0001-initial-architecture.md)
