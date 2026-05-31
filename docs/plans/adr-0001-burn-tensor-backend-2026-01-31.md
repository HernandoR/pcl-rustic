# ADR-0001: Burn Tensor Framework as Computation Backend

- Status: Accepted
- Date: 2026-01-31

## Context

The project needs a high-performance computation backend for point cloud
operations. Point clouds can contain millions of points, each with multiple
attributes (XYZ, intensity, RGB, custom). Operations like voxel downsampling
and coordinate transforms must process all points in bulk.

Key requirements:
- Batch vectorized operations on tensors (no per-point loops in Python)
- CPU support required, GPU support desirable for future
- Memory-contiguous storage for cache efficiency
- Integration with Rust's type system and PyO3 for Python bindings

Alternatives considered:
- **ndarray**: Rust-only, no GPU path, but simpler dependency
- **tch-rs (libtorch)**: Heavy C++ dependency, complex build
- **candle**: Lighter than libtorch but less mature
- **Burn**: Pure Rust, supports multiple backends (CPU/GPU), growing ecosystem

## Decision

Use the **Burn** tensor framework (`burn` crate) as the sole computation
backend for all point cloud data storage and operations.

Internal storage types (see `src/point_cloud/core.rs:11-25`):

```rust
pub struct HighPerformancePointCloud {
    xyz: Tensor2,                          // [M, 3] Burn tensor
    intensity: Option<Tensor1>,            // [M] optional
    rgb_r: Option<Tensor1>,                // 3 separate channels
    rgb_g: Option<Tensor1>,
    rgb_b: Option<Tensor1>,
    attributes: HashMap<String, Tensor1>,  // custom attributes
}
```

Utility layer at `src/utils/tensor.rs` provides helpers for tensor creation,
validation, and conversion to/from `Vec` types.

## Consequences

**Positive:**
- All operations are batch-vectorized through Burn's tensor ops
- Memory-contiguous storage improves cache locality
- Single dependency for computation — no mixing of ndarray + GPU lib
- GPU backend can be enabled later by swapping the Burn backend feature flag
- Pure Rust — no C++ build dependency (unlike libtorch)

**Negative:**
- Burn's API surface and versioning are still evolving
- Tensor-to-Vec conversions incur serialization cost at Python boundary
  (mitigated by zero-copy numpy interop at the PyO3 layer)
- Internal tensors use `f32` exclusively — no mixed-precision support
