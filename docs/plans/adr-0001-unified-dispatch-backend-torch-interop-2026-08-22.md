# ADR-0001: Unified Dispatch Backend & Torch Interop

- Status: Accepted
- Date: 2026-08-22
- Provenance: RFC-0001 §2.1, §2.3; burn issue tracel-ai/burn#4415

## Context

burn resolved runtime backend selection (issue #4415, filed by the project
owner) in 0.21: compile several backends, pick one per tensor via a device
value. The pre-redesign code pinned burn 0.20.1 and hand-rolled
`Router<(Wgpu, NdArray)>` with a custom auto-detect helper. The owner also
wants torch-tensor compatibility for Python users.

## Decision

### Backend

Use **burn 0.21** with the unified dispatch backend:

```rust
// src/backend.rs
pub type B = burn::Dispatch;              // backend type in every tensor signature
pub use burn::DispatchDevice;             // runtime device selection
```

Cargo features (crate → burn passthrough):

| crate feature | burn feature | default | provides |
|---|---|---|---|
| `cpu` | `flex` | yes | pure-Rust CPU device (see note) |
| `vulkan` | `vulkan` | yes | wgpu/Vulkan GPU device |
| `cuda` | `cuda` | no | native CUDA device |
| `torch` | `tch` | no | LibTorch device (tensors are torch tensors) |

burn is depended on with `default-features = false` and
`features = ["std", "dispatch"]`. wgpu flavors are mutually exclusive in
burn-dispatch; `vulkan` is our pick for Linux/Windows portability. The
0.20-era `candle-core` pin and the `ndarray` feature are dropped.

`cpu` is backed by `burn/flex`, not the CubeCL `burn/cpu` backend: the
latter compiles cleanly but aborts at runtime on tall matmuls. A
`[N,3] × [3,3]` f32 matmul on `DispatchDevice::Cpu` overflows the stack of
an internal cubecl worker thread for N ≥ ~5.6M (N = 5M completes in 2.3s);
growing the caller's stack does not help, because the overflowing thread is
spawned internally. The same failure reproduces on burn 0.21.0 and
0.22.0-pre.2, and `flex` runs the identical [10M,3] workload in 0.22s.
Upstream issue: <https://github.com/tracel-ai/burn/issues/5419>. Revisit when
burn-cpu stabilizes.

The `torch` feature additionally declares a direct optional dependency on
`burn-dispatch` with its `tch` feature, because burn 0.21's `tch` feature is
the only backend feature that does not forward to `burn-dispatch`; without
it, `DispatchDevice::LibTorch` is never compiled. Upstream fixed this in
0.22 (`burn/tch` → `burn-core/tch` → `burn-tensor/tch` = `burn-dispatch/tch`),
so the phantom dependency is dropped when we move to 0.22.

### Device selection

- `default_device()` probes at runtime: GPU variant if an adapter initializes,
  else CPU. Probing failure degrades gracefully — never a panic at import.
- Python: `pcl_rustic.available_devices() -> list[str]`,
  `pc.device -> str`, `pc.to_device("cpu" | "vulkan" | "cuda" | "torch")`.
  Names map 1:1 to `DispatchDevice` variants compiled in.

### Torch-tensor compatibility

Two complementary planes (RFC-0001 §2.3):

1. **Boundary (always on):** every array-accepting API takes
   `numpy.ndarray | torch.Tensor | buffer-protocol object`. Torch tensors are
   ingested zero-copy on CPU via DLPack (`np.from_dlpack`); CUDA torch
   tensors are accepted with an explicit device→host copy. Export:
   `pc.to_torch(name)` (zero-copy `torch.from_numpy` where dtypes allow).
   torch is an optional import — nothing in pcl-rustic requires it.
2. **Backend (opt-in build):** the `torch` cargo feature compiles
   `burn-tch`, adding a `LibTorch` dispatch device so burn ops execute on
   libtorch (including its CUDA allocator).

### What burn is, and is not, used for

*Added 2026-08-23 after profiling; see RFC-0001.* `Dispatch` is the device
and tensor abstraction, not a mandate that every kernel go through burn.
Where a hand-written kernel is materially faster on CPU, we use it:

- Coordinate transforms on CPU-resident clouds run a rayon `R * p + t` loop
  in f64 (`cloud/transform.rs`), not a tensor matmul. For the degenerate
  `[N,3] x [3,3]` shape a tiled matmul kernel wins nothing: measured at 10M
  points, burn needs 173-273 ms where the plain loop needs 23 ms.
- Voxel binning and LAS decoding are likewise plain Rust over rayon.

burn owns device-resident data and every operation on it, which is what the
GPU story depends on. The rule of thumb: tensor ops for anything that must
run on an accelerator, hand-written kernels for CPU-only hot paths where a
measurement justifies it.

## Consequences

**Positive:** one wheel serves GPU-less and GPU hosts (the exact #4415
motivation); no custom router plumbing to maintain; torch users get zero-copy
CPU interop by default and torch-native execution when they build with
`torch`; `Dispatch` in signatures removes the generic `<B: Backend>`
plumbing.

**Negative:** burn 0.21 is a fast-moving dependency (dispatch is new); the
`torch` build inherits libtorch version pinning from `tch`; per-tensor
dispatch adds a small runtime indirection over a monomorphized backend —
irrelevant next to point-cloud batch sizes.
