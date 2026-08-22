# ADR-0002: LAS-Aligned Dimension & Dtype Contract

- Status: Accepted
- Date: 2026-08-22
- Provenance: RFC-0001 §2.2, §2.4

## Context

The pre-redesign contract was ad-hoc: `from_xyz` accepted four dtypes,
attribute setters demanded `float32`, RGB was f32 internally and u8
externally, and LAS fields (classification u8, GPS time f64, intensity u16)
were lossily coerced. The project owner mandated adopting the dtype promise
laspy makes for point records, so a laspy user can move to pcl-rustic without
changing dtypes.

## Decision

### Standard dimensions

A `PointCloud` is a set of named, equal-length dimensions. Standard dimension
names carry **pinned numpy dtypes**, identical to laspy's:

| Dimension | numpy dtype | Notes |
|---|---|---|
| `x`, `y`, `z` | `f8` | scaled coordinates (see storage below) |
| `intensity` | `u2` | |
| `return_number` | `u1` | unpacked from LAS bit field |
| `number_of_returns` | `u1` | unpacked |
| `synthetic`, `key_point`, `withheld`, `overlap` | `bool` | unpacked flags |
| `scan_direction_flag`, `edge_of_flight_line` | `bool` | unpacked |
| `scanner_channel` | `u1` | formats 6–10 |
| `classification` | `u1` | |
| `user_data` | `u1` | |
| `scan_angle` | `i2` | LAS 0–5 `scan_angle_rank` (i1) widens losslessly |
| `point_source_id` | `u2` | |
| `gps_time` | `f8` | |
| `red`, `green`, `blue` | `u2` | 16-bit color, per LAS — **not** u8 |
| `nir` | `u2` | |

Deviations from laspy, both deliberate: bit fields are stored unpacked (we
are an in-memory compute structure, not a file codec), and `scan_angle_rank`
is normalized to `scan_angle` i2 on read.

Waveform dimensions (formats 4/5/9/10) are reserved but unimplemented
(RFC-0001 §4).

### Extra dimensions

`add_extra_dim(name, dtype)` accepts exactly laspy's allowed scalar set:
`u1 u2 u4 u8 i1 i2 i4 i8 f4 f8`. The declared dtype is preserved end-to-end,
including LAS ExtraBytes and Parquet round-trips (ADR-0004).

### Getter / setter semantics

- Getters return a numpy array of the pinned dtype. Dimension access is
  uniform: `pc["intensity"]`, plus attribute sugar (`pc.intensity`) for
  standard names.
- Setters accept an exact-dtype array as-is; other dtypes are coerced with
  numpy `same_kind` casting. Value outside the target range raises
  `OverflowError`; kind-incompatible input (e.g. float array into `u1`)
  raises `TypeError`. This mirrors laspy's assignment behavior.
- `x/y/z` setters accept any real dtype and store via the coordinate path.

### Coordinate storage

`x/y/z` are **promised** as f8 but **stored** as LAS itself stores them:

```
coords_rel : Tensor<Dispatch, 2> (f32, on the compute device)  # [N, 3]
offset     : [f64; 3]                                          # per-cloud anchor
scales     : [f64; 3]                                          # kept for LAS write fidelity
```

- Getter: `x = offset[0] + f64(coords_rel[:, 0])`.
- Ingest: offset defaults to the first point (LAS ingest uses the header
  offset), so relative magnitudes stay small and f32 error stays below LAS
  quantization for typical extents.
- Translation mutates only `offset` (exact f64); rotation/matmul runs on the
  f32 tensor.

### Attribute storage

Non-coordinate dimensions live CPU-side in a dtype-tagged column store
(`DimArray` enum over `u8/u16/u32/u64/i8/i16/i32/i64/f32/f64/bool` vectors),
ordered by insertion. They do not ride the tensor device until a GPU op needs
them (open question in RFC-0001 §5).

## Consequences

**Positive:** laspy users keep their dtypes; LAS files round-trip losslessly;
exact integer equality works for classification-based selection (future
selection ops); GPU compute stays f32 without breaking the f8 promise.

**Negative:** getters materialize (copy) output arrays — the zero-copy of a
raw f32 view is gone by design, since promised dtypes differ from storage;
RGB consumers expecting u8 must shift (`>> 8`) themselves; two storage planes
(tensor coords vs CPU columns) must stay length-synchronized, enforced by a
single mutation choke point in `cloud/mod.rs`.
