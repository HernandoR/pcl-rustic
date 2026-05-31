# ADR-0008: Multi-Format File I/O with Column-Name Mapping

- Status: Accepted
- Date: 2026-01-31

## Context

Point cloud data arrives in multiple file formats: LAS/LAZ (industry standard
for LiDAR), CSV (text-based exchange), and Parquet (columnar analytics). Each
format has different internal structure, compression, and column naming
conventions. The API should provide per-format methods for explicit control
and convenience methods for automatic format detection.

## Decision

Provide **per-format static/instance methods** on `PointCloud` plus
**auto-detect convenience methods**. File I/O is implemented as direct methods
on `HighPerformancePointCloud`, not through a trait.

Format-specific methods (`src/io/`):

| Format | Read | Write | Module |
|--------|------|-------|--------|
| LAS/LAZ | `from_las(path)` | `to_las(path, compress)` | `io/las_laz.rs` |
| CSV | `from_csv(path, delimiter, columns...)` | `to_csv(path, delimiter, columns...)` | `io/csv.rs` |
| Parquet | `from_parquet(path, columns...)` | `to_parquet(path, columns...)` | `io/parquet.rs` |

Convenience methods (auto-detect by extension):
- `load_from_file(path, ...)` — dispatches to the correct format reader
- `save_to_file(path, ...)` — dispatches by file extension

Column-name mapping (`src/io/table.rs`): `TableColumnNames::resolve()` maps
optional column name strings (`x`, `y`, `z`, `intensity`, `rgb_r`, `rgb_g`,
`rgb_b`) to a structured config. When column names are `None`, sensible
defaults are used (e.g., `"x"`, `"y"`, `"z"`).

CSV delimiter is exposed as a `u8` (ASCII byte value) — not a character or
string — because the `csv` crate and the PyO3 binding both use byte delimiters.
Common values: `44` (comma), `9` (tab).

Against: using a unified `IOConvert` trait (adds abstraction without benefit —
each format has different parameters that don't map cleanly to a single
interface), or using only `load_from_file`/`save_to_file` (loses per-format
control over compression, delimiter, etc.).

## Consequences

**Positive:**
- Per-format methods allow format-specific parameters (LAS compression,
  CSV delimiter, Parquet column names)
- `load_from_file`/`save_to_file` provide a simple default path for common cases
- Adding a new format means: add a module in `src/io/`, add methods on
  `HighPerformancePointCloud`, add PyO3 bindings — no trait to implement
- Column-name mapping is explicit and configurable per call

**Negative:**
- No trait means each new format must wire up PyO3 bindings manually
- The PyO3 binding layer has 8 I/O methods (`from_las`, `to_las`, `from_csv`,
  `to_csv`, `from_parquet`, `to_parquet`, `load_from_file`, `save_to_file`)
  with similar but not identical signatures
- CSV delimiter as `u8` integer (44) is non-obvious — Python users expect
  `","` or `b","`; mitigated by documentation
- `delete_file` is a static method on `PointCloud` rather than an OS-level
  utility, coupling file management to the point cloud class
