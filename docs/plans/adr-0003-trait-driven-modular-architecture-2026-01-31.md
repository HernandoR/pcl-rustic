# ADR-0003: Trait-Driven Modular Architecture

- Status: Accepted
- Date: 2026-01-31

## Context

The Rust codebase needs a modular design that separates concerns, enables
independent testing, and supports future extension without modifying core
structures. Point cloud operations span multiple domains: data access,
attribute management, coordinate transforms, downsampling, and file I/O.

The project structure splits across 5 modules (`src/traits/`, `src/point_cloud/`,
`src/io/`, `src/interop/`, `src/utils/`) with 21 source files.

## Decision

Define **5 traits** in `src/traits/` that together form the API contract, with
`HighPerformancePointCloud` implementing 4 of them:

| Trait | File | Purpose | Implementor |
|-------|------|---------|------------|
| `PointCloudCore` | `traits/point_cloud.rs:6` | Read XYZ, intensity, RGB, attributes | `HighPerformancePointCloud` |
| `PointCloudProperties` | `traits/point_cloud.rs:34` | Set/remove intensity, RGB, custom attributes | `HighPerformancePointCloud` |
| `CoordinateTransform` | `traits/transform.rs:4` | 3×3 and 4×4 matrix transforms, rigid transform | `HighPerformancePointCloud` |
| `VoxelDownsample` | `traits/downsample.rs:14` | Voxel downsampling with pluggable strategy | `HighPerformancePointCloud` |
| `DownsampleStrategy` | `traits/downsample.rs:5` | Strategy interface for representative point selection | `RandomSampleStrategy`, `CentroidSampleStrategy` |

Implementation modules are organized by concern:
- `src/point_cloud/core.rs` — struct definition and core trait impls
- `src/point_cloud/voxel.rs` — `VoxelDownsample` impl + strategy structs
- `src/point_cloud/transform.rs` — `CoordinateTransform` impl
- `src/point_cloud/attributes.rs` — attribute helper methods
- `src/io/` — file format implementations (direct methods, no I/O trait)
- `src/interop/numpy.rs` — NumPy array conversion
- `src/utils/` — error types, tensor helpers, reflection utilities

File I/O is **not** behind a trait — each format (`las_laz.rs`, `csv.rs`,
`parquet.rs`) provides standalone functions called directly by
`HighPerformancePointCloud` methods and the PyO3 binding layer.

## Consequences

**Positive:**
- New operations can be added by defining a new trait and implementing it
- Existing traits are isolated — changing `CoordinateTransform` does not risk
  breaking `VoxelDownsample`
- Strategy structs implement `DownsampleStrategy` independently, enabling
  compile-time dispatch
- Traits document the API contract at the Rust level

**Negative:**
- 5 traits for a single implementor (`HighPerformancePointCloud`) may be
  over-abstraction — traits are not reused across types
- File I/O is deliberately not behind a trait, creating an inconsistency in
  the design (some functionality is trait-based, some is direct)
- `PointCloudCore` and `PointCloudProperties` could arguably be one trait
  but are split for interface segregation
