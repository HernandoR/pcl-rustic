# ADR-0004: Multi-Format File I/O

- Status: Accepted
- Date: 2026-08-22
- Provenance: RFC-0001 §3 (item 4)

## Context

Point cloud data arrives in multiple file formats: LAS/LAZ (industry standard
for LiDAR), CSV (text-based exchange), and Parquet (columnar analytics).
Since ADR-0002, dimensions are typed (laspy-aligned dtypes), so I/O must
preserve column dtypes, not funnel everything through f32 as the
pre-redesign code did.

## Decision

Two laspy-style entry points on `PointCloud`, dispatching on file extension:

```python
pc = PointCloud.read(path, **options)   # .las .laz .csv .tsv .parquet
pc.write(path, **options)               # .laz implies compression
```

Format modules (`src/io/`):

| Format | Module | Backing crate | Dtype fidelity |
|--------|--------|---------------|----------------|
| LAS/LAZ | `io/las.rs` | `las` (laz-parallel) | standard dims per ADR-0002 table; see limitations below |
| CSV | `io/csv.rs` | `csv` | header-driven (`.csv`/`.tsv`); standard names adopt pinned dtypes, others parse as f8 |
| Parquet | `io/parquet.rs` | `arrow` + `parquet` | exact Arrow↔dtype mapping both ways |

polars is dropped: it was a heavy dependency used only as a CSV/Parquet shim,
and its DataFrame types forced an extra copy. arrow record batches map 1:1
onto the typed column store.

Options:
- `columns: dict[str, str]` — maps dimension names to file column names for
  CSV/Parquet; writes use it directly, reads invert it (default: identity,
  with `x/y/z` required).
- `delimiter: str` (single ASCII char, default `","`) — CSV only.
- LAZ compression is implied by the `.laz` extension; no `compress` flag.

Known `las`-crate (0.9) limitations, accepted for v1 and documented in
`src/io/las.rs`:
- Extra dimensions are not written as LAS ExtraBytes (the crate exposes no
  typed ExtraBytes API); Parquet is the lossless carrier for them.
- `scan_angle` is reconstructed from the crate's f32 degrees representation.
- Classification code 12 is rejected by the crate (reserved legacy overlap
  encoding); it surfaces as a `ValueError` rather than silent coercion.

Against: per-format method pairs (`from_las`/`to_las`, `from_csv`/…) — eight
near-identical bindings whose differences are better expressed as kwargs; a
unified `IOConvert` trait — abstraction without a second consumer.

## Consequences

**Positive:**
- One read and one write entry point mirrors `laspy.read` / `las.write`.
- Typed columns round-trip losslessly through Parquet; standard LAS
  dimensions round-trip losslessly through LAS/LAZ.
- Adding a format = one module in `src/io/` + one match arm in `io/mod.rs`.

**Negative:**
- Extension-based dispatch means a misnamed file needs a rename (or a future
  `format=` override) — acceptable for v1.
- CSV cannot express dtypes; non-standard columns come back as f8 unless the
  caller re-declares them (documented).
