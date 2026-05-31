# ADR-0004: Private Fields with Batch-Only Access

- Status: Accepted
- Date: 2026-01-31

## Context

Point cloud data is inherently bulk data — operations on individual points are
rarely useful and would be prohibitively slow across the Python/Rust boundary.
The system must prevent users from accidentally accessing or modifying single
points, which would defeat batch optimizations and cause excessive
serialization overhead.

## Decision

All fields of `HighPerformancePointCloud` are **private**. Access is provided
only through trait methods and public APIs that operate on entire arrays.

Internal crate-only access uses `pub(crate)` reference getters:

```rust
// src/point_cloud/core.rs
pub(crate) fn xyz_ref(&self) -> &Tensor2 { &self.xyz }
pub(crate) fn xyz_mut(&mut self) -> &mut Tensor2 { &mut self.xyz }
pub(crate) fn intensity_ref(&self) -> Option<&Tensor1> { ... }
pub(crate) fn rgb_channels_ref(&self) -> (Option<&Tensor1>, ...) { ... }
pub(crate) fn attributes_ref(&self) -> &HashMap<String, Tensor1> { ... }
```

The Python API exposes only batch operations:
- `get_xyz()` → `np.ndarray` shape `[N, 3]`
- `get_intensity()` → `np.ndarray` shape `[N]`
- `set_intensity(arr)` — takes full `[N]` array
- No `get_point(i)` or `set_point(i, x, y, z)` methods exist

Against: exposing individual field accessors, point-level getters/setters,
or making fields `pub`.

## Consequences

**Positive:**
- Enforces batch processing — impossible to accidentally loop over points in
  Python
- Rust compiler guarantees no external code can bypass the trait methods
- Internal refactoring of field types (e.g., changing `Tensor2` implementation)
  is invisible to consumers
- Memory safety: no way to create dangling references to individual elements

**Negative:**
- Debugging requires going through the trait methods or adding temporary
  `pub(crate)` accessors
- Users coming from libraries like Open3D (which support point-level access)
  may find the constraint unfamiliar
- Test code inside `src/point_cloud/` must use the `pub(crate)` reference
  getters rather than direct field access
