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

*Amended 2026-08-23 after profiling; see RFC-0001.* Coordinates have two
representations, chosen by device placement:

```
// CPU-resident (the common case)
coords : Vec<f64>            // absolute, row-major [N, 3]

// Device-resident (wgpu/CUDA have no f64)
coords : Tensor<Dispatch, 2> // f32, relative to `offset`
offset : [f64; 3]            // per-cloud anchor, also the LAS header origin
scales : [f64; 3]            // kept for LAS write fidelity
```

- **CPU:** `x/y/z` are stored exactly as promised, f64. Reads are a copy of
  the buffer, with no conversion. `x` is bit-identical to what laspy returns
  for the same file.
- **Device:** `x = offset[0] + f64(coords[:, 0])`. `offset` anchors near the
  data (ingest uses the first point; LAS ingest uses the header offset), so
  f32 error stays below LAS quantization for typical extents.
- Moving between devices re-encodes. GPU -> CPU widens back to f64 but does
  not recover rounding already incurred.
- Transforms: CPU runs a specialised `R * p + t` kernel over rayon in f64
  (not a matmul -- see RFC-0001); device applies the rotation as an f32
  tensor matmul. In both cases `offset` moves as `R * offset + t` in f64.

The original design stored f32-relative coordinates unconditionally. That
was measured to be the dominant cost of every coordinate read (an f32 -> f64
widening pass over the whole cloud) *and* the source of a 3.03e-5 m deviation
from laspy. Holding f64 on CPU removed both at once.

*Amended 2026-08-29: configurable coordinate dtype.* Once the Vulkan backend
actually initialized (it had been silently failing; see the device_report
work), the widening pass returned as the dominant cost of GPU readback
(~250 ms of a 260 ms `.xyz` at 10M points). Coordinates now carry a
process-wide default dtype, torch-style (`set_default_dtype`,
`PCL_RUSTIC_DEFAULT_DTYPE`), fixed per cloud at construction:

- **f32 (default):** storage is f32-relative-to-offset on CPU *and* GPU (the
  representations are identical, so device moves copy the buffer as-is), and
  reads return float32 -- no widening pass, half the copy volume. Absolute
  values quantize to the f32 grid at readout (extent-dependent; ~0.25 m at
  raw UTM magnitude), while internal anchoring, transforms, and LAS writes
  stay f64-exact through the offset.
- **f64 (opt-in):** the pre-amendment behavior -- absolute f64 on CPU,
  bit-identical to laspy. f64 clouds ingest CPU-resident even under a GPU
  default device, because the device representation is f32 and honoring
  GPU-first placement would round at construction; rounding under f64 only
  happens through an explicit `to_device`.

The "x/y/z are f8" row of the dtype table above becomes the *setter*
contract (writes are always accepted as f8); the getter dtype follows the
cloud's coordinate dtype.

*Amended 2026-08-29 (2): zero-copy coordinate views.* The CPU-f64 buffer
moved behind an `Arc` (copy-on-write), and `PointCloud.view("xyz"|"x"|"y"|
"z")` returns read-only numpy views over it -- zero-copy, base-tied to the
allocation, stable snapshots under later mutation (mutating paths go through
`Arc::make_mut`). This relaxes "getters always copy" for this one explicit,
opt-in API; `.xyz` and friends keep copying semantics. f32 clouds raise:
their storage is offset-relative, so readout computes and cannot be a view.
`PointCloud::clone()` incidentally became O(1) on f64 coordinates.

### Attribute storage

Non-coordinate dimensions live CPU-side in a dtype-tagged column store
(`DimArray` enum over `u8/u16/u32/u64/i8/i16/i32/i64/f32/f64/bool` vectors),
ordered by insertion. They do not ride the tensor device until a GPU op needs
them (open question in RFC-0001 §5).

## Consequences

**Positive:** laspy users keep their dtypes; LAS files round-trip losslessly,
coordinates included and bit-exact on CPU; exact integer equality works for
classification-based selection (future selection ops); GPU compute still gets
f32 without breaking the f8 promise.

**Negative:** two coordinate representations exist, and code touching them
must handle both (confined to `cloud/mod.rs`, `cloud/transform.rs`, and
`cloud/voxel.rs`); a CPU cloud costs 8 bytes per ordinate rather than 4;
coordinate getters still copy, because handing out a borrowed view would
dangle as soon as a setter replaced the buffer (a copy-on-write `Arc` would
fix this and is deferred); RGB consumers expecting u8 must shift (`>> 8`)
themselves; the two planes (coords vs columns) must stay length-synchronized,
enforced by a single mutation choke point in `cloud/mod.rs`.
