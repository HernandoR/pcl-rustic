# ADR-0007: Multi-Dtype NumPy Interop

- Status: Accepted
- Date: 2026-01-31

## Context

The Python API must accept NumPy arrays as input and return NumPy arrays as
output. Users may provide data in various dtypes (float32, float64, int32,
int64) depending on their data pipeline. The system should be lenient on input
(for user convenience) but strict where conversion would be ambiguous.

## Decision

**from_xyz supports 4 dtypes** — `float32`, `float64`, `int32`, `int64`.
Conversion to internal `f32` tensors happens automatically via
`read_xyz_from_pyany()` in `src/interop/numpy.rs:74`.

**Attribute setters require float32** — `set_intensity`, `set_rgb`, 
`add_attribute`, `set_attribute` all use `read_attribute_array()` which casts
to `PyArray1<f32>` and rejects other dtypes with a clear error message
(`src/lib.rs:588-600`).

Output dtypes are fixed:
- `get_xyz()` → `float32`
- `get_intensity()` → `float32`
- `get_rgb()` → `uint8` (see ADR-0006)

Implementation detail (`src/interop/numpy.rs`):
- `from_xyz_array()` reads via `PyUntypedArrayMethods` then converts
- `from_numpy()` handles dictionary-based construction with optional keys
- `to_numpy()` produces a flat dict with `xyz`, optional `intensity`,
  `r`/`g`/`b`, and custom attributes

Against: requiring `float32` for all inputs (too strict — users with float64
pipelines would need explicit casts everywhere), or supporting all numpy dtypes
for attribute setters (risk of silent precision loss).

## Consequences

**Positive:**
- `from_xyz` is lenient — users can pass common numeric dtypes without
  pre-conversion
- Attribute setters are strict — precision loss requires an explicit
  `.astype(np.float32)`, making the conversion visible
- Zero-copy is achieved for `float32` inputs (the common case)
- Error messages are explicit: "必须是dtype=float32的1D numpy数组，请使用
  arr.astype(np.float32) 转换"

**Negative:**
- Asymmetric: `from_xyz` accepts 4 dtypes but attribute setters accept only 1
  — users may be surprised when `set_intensity(int64_array)` fails
- `int32`/`int64` → `f32` conversion may lose precision for large integers
  (> 2^24), though this is uncommon in point cloud coordinates
- The dtype checking happens at the PyO3 boundary, not at the Rust trait level
  — pure Rust callers could pass any `Vec<f32>` without dtype checks
