# ADR-0002: PyO3 Single-Class Python Binding

- Status: Accepted
- Date: 2026-01-31

## Context

The Python API must expose Rust point cloud functionality to Python users in a
way that feels idiomatic. Users should interact with one primary class, not
navigate multiple Rust structs exposed directly.

Key concerns:
- Python users expect a `PointCloud` class with methods, not module-level functions
- Need to map Rust's trait-based design to Python's class-based model
- Error types must translate cleanly from Rust to Python exceptions
- IDE autocomplete and type checking should work via `.pyi` stubs

## Decision

Expose a **single Python class** `PointCloud` (backed by `PyPointCloud` in Rust)
that wraps `HighPerformancePointCloud` and delegates all operations to it.
Strategy selection uses a separate `DownsampleStrategy` enum class with integer
class attributes.

Implementation (`src/lib.rs:26-29`):

```rust
#[pyclass(name = "PointCloud")]
pub struct PyPointCloud {
    inner: HighPerformancePointCloud,
}
```

All 30+ public methods are defined on `PyPointCloud` via `#[pymethods]`. The
inner `HighPerformancePointCloud` is never exposed to Python — it remains an
opaque Rust struct.

Strategy mapping (`src/lib.rs:563-579`):

```rust
#[pyclass(name = "DownsampleStrategy")]
pub struct PyDownsampleStrategy;

#[pymethods]
impl PyDownsampleStrategy {
    #[classattr] fn RANDOM() -> i32 { 0 }
    #[classattr] fn CENTROID() -> i32 { 1 }
}
```

Error conversion is handled by `From<PointCloudError> for PyErr`
(`src/utils/error.rs:47`).

Against: exposing multiple Python classes (one per Rust module), or using
module-level functions instead of a class.

## Consequences

**Positive:**
- Python API is simple: `from pcl_rustic import PointCloud, DownsampleStrategy`
- Single import covers all functionality
- Type stubs can describe the full API in one `.pyi` file
- Opaque inner struct prevents Python code from depending on Rust internals

**Negative:**
- `PyPointCloud` is a large class with 30+ methods — harder to navigate
- Strategy enum uses raw `i32` values rather than true Python enums,
  requiring manual mapping in the `voxel_downsample` match block
- Adding a new strategy requires editing both the Rust match block and the
  Python class attribute
