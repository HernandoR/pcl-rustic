# ADR-0006: Three-Channel RGB Storage (f32 Internal, u8 External)

- Status: Accepted
- Date: 2026-01-31

## Context

RGB color data in point clouds is typically stored as 8-bit unsigned integers
per channel (0–255). However, Burn tensors operate on `f32` for computation.
The design must reconcile internal computation needs with external data format
conventions.

## Decision

Store RGB as **three separate `Option<Tensor1>` fields** internally as `f32`,
but return them as `u8` to external consumers.

Internal storage (`src/point_cloud/core.rs:18-21`):

```rust
pub struct HighPerformancePointCloud {
    // ...
    rgb_r: Option<Tensor1>,  // f32 internally
    rgb_g: Option<Tensor1>,
    rgb_b: Option<Tensor1>,
}
```

The `PointCloudCore` trait returns `u8`:

```rust
fn get_rgb(&self) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)>;
```

The `PointCloudProperties` trait accepts `u8`:

```rust
fn set_rgb(&mut self, r: Vec<u8>, g: Vec<u8>, b: Vec<u8>) -> Result<()>;
```

Conversion between `f32` tensors and `u8` vectors happens at the trait
boundary using helper functions in `src/utils/tensor.rs` (`tensor1_to_u8_vec`).

The Python API mirrors this:
- `get_rgb()` returns 3 `uint8` numpy arrays
- `set_rgb(r, g, b)` accepts 3 `float32` numpy arrays (converted to u8 internally)

Against: storing RGB as a single `Tensor2` `[M, 3]` (would require reshaping
for per-channel operations), or using `u8` tensors directly (Burn primarily
operates on `f32`).

## Consequences

**Positive:**
- Internal tensors are homogeneous `f32` — all tensor operations
  (matmul, add, etc.) work without type casting
- Three separate channels allow independent access to R, G, B
- External API returns standard `u8` — compatible with visualization tools
- `rgb_r`, `rgb_g`, `rgb_b` can independently be `None` (though in practice
  they are set/cleared together)

**Negative:**
- Three `Option<Tensor1>` fields instead of one `Option<Tensor2>` increases
  struct field count
- Setting RGB requires 3 separate array arguments in Python (`set_rgb(r, g, b)`)
  rather than a single `[N, 3]` array
- Conversion at every get/set boundary adds overhead for RGB-heavy workflows
- Memory: 4 bytes per channel internally vs. 1 byte in external format
