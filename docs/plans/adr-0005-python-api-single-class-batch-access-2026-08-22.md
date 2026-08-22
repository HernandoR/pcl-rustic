# ADR-0005: Python API Shape — Single Class, Batch-Only Access

- Status: Accepted
- Date: 2026-08-22
- Provenance: RFC-0001 §3 (item 5)

## Context

The Python surface needs a shape decision independent of the dtype contract:
how many classes, how mutation is exposed, and where the numpy/torch boundary
logic lives. Single-point editing was explicitly ruled out by the project
owner in the pre-redesign roadmap and that constraint carries over.

## Decision

- **One public class.** `pcl_rustic.PointCloud` is the entire object surface,
  plus the `DownsampleStrategy` constants holder and module-level functions
  (`read`, `available_devices`, `default_device`).
- **Composition over the extension class.** The PyO3 `_core.PointCloud`
  demands exact dtypes and stays minimal; the Python `PointCloud` wraps it by
  composition (`_inner`) and owns all convenience: dtype coercion
  (ADR-0002 semantics), torch/DLPack ingestion (ADR-0001), attribute sugar
  (`pc.intensity`), item access (`pc["classification"]`), and rewrapping of
  returned cores. Rationale: numpy casting rules are native to Python;
  keeping them out of Rust halves the binding surface; PyO3 static
  constructors return the base class, which makes subclassing brittle.
- **Batch-only access.** All getters/setters move whole dimensions
  (`ndarray` in, `ndarray` out). There is no point indexing, no per-point
  proxy object, and no in-place scalar mutation. Geometry operations
  (`transform`, `rigid_transform`, `voxel_downsample`, `to_device`) return
  new `PointCloud` instances.

## Consequences

**Positive:** one import, one class to learn; the Rust boundary stays small
and exact-typed (every fuzzy input problem is handled once, in Python);
batch-only access keeps every operation vectorizable and GPU-eligible.

**Negative:** every `_core` method needs a thin delegation stub in the
wrapper (~20 boring lines); users wanting `pc[i]` single-point access must
slice arrays themselves; two objects exist per cloud (wrapper + core), which
costs one indirection per call — negligible against batch workloads.
