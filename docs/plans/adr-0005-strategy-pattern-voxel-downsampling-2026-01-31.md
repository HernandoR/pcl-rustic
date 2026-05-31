# ADR-0005: Strategy Pattern for Voxel Downsampling

- Status: Accepted
- Date: 2026-01-31

## Context

Voxel downsampling needs to select one representative point per voxel cell.
Different applications need different selection criteria: speed (random),
geometric accuracy (centroid), or attribute-weighted selection. The selection
logic should be pluggable without modifying the core downsampling algorithm.

## Decision

Use the **Strategy pattern** via the `DownsampleStrategy` trait. The
`VoxelDownsample` trait accepts a `Box<dyn DownsampleStrategy>` and delegates
point selection to it.

Trait definition (`src/traits/downsample.rs:5`):

```rust
pub trait DownsampleStrategy: Send + Sync {
    fn select_representative(
        &self,
        indices: Vec<usize>,
        xyz: &[Vec<f32>],
    ) -> Result<usize>;
}
```

Two strategies are implemented (`src/point_cloud/voxel.rs`):

```rust
pub struct RandomSampleStrategy;     // selects middle index of the voxel
pub struct CentroidSampleStrategy;   // selects point closest to voxel center
```

The `VoxelDownsample` implementation in `voxel.rs:32`:
1. Groups points by voxel using coordinate quantization (O(N) via `reflect.rs`)
2. Iterates over voxel groups, calling `strategy.select_representative()`
3. Builds a new `HighPerformancePointCloud` from selected indices, preserving
   all attributes (intensity, RGB, custom)

Python binding maps integer strategy IDs to boxed trait objects
(`src/lib.rs:321-326`):

```rust
fn voxel_downsample(&self, voxel_size: f32, strategy: i32) -> PyResult<Self> {
    let strategy_impl: Box<dyn DownsampleStrategy> = match strategy {
        0 => Box::new(RandomSampleStrategy),
        1 => Box::new(CentroidSampleStrategy),
        _ => return Err(...),
    };
    self.inner.voxel_downsample(voxel_size, strategy_impl)
}
```

## Consequences

**Positive:**
- Adding a new strategy requires only: (a) implement `DownsampleStrategy`,
  (b) add a match arm in `lib.rs`, (c) add a class attribute on
  `PyDownsampleStrategy`
- Core downsampling algorithm (voxel grouping, index collection, point cloud
  reconstruction) is unchanged regardless of strategy
- Strategies are compile-time checked to implement the trait

**Negative:**
- `Box<dyn DownsampleStrategy>` uses dynamic dispatch — there is a vtable
  lookup per voxel group (mitigated because the number of voxels is much
  smaller than the number of points)
- Python strategy IDs are raw integers; there is no compile-time guarantee
  that all valid IDs map to a strategy
- Strategies receive `Vec<usize>` by value (clone per voxel group), which
  adds allocation overhead
