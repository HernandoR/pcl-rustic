# RFC-0001: Full-Structure Redesign

- **Status:** Resolved (→ ADR-0001 … ADR-0005)
- **Date:** 2026-08-22
- **Author:** agent session with project owner (HernandoR)
- **Related:** burn issue
  [tracel-ai/burn#4415](https://github.com/tracel-ai/burn/issues/4415),
  laspy point records
  (<https://laspy.readthedocs.io/en/latest/intro.html#point-records>)

This RFC starts a fresh record set. The pre-redesign RFCs and ADRs were
removed at the project owner's direction (2026-08-22) to avoid confusion:
they described an API and architecture that no longer exist. Their history
remains in git before this commit.

## 1. Problem

Five structural problems were raised by the project owner on 2026-08-22, with
an explicit "no backward compatibility" mandate:

1. **RFCs and ADRs shared one folder** (`docs/plans/`), mixing append-only
   discussion logs with atomically-maintained decision records, so neither
   convention could be enforced.
2. **The Python test/benchmark environment was not simple.** Concretely:
   `pyproject.toml` pointed `readme` at a deleted `ai_doc/README.md`, so
   *every* `uv run` failed to build the editable wheel; `pytest.ini` forced
   coverage + junit + `-s` on every run; benchmarks were addressed by a fully
   qualified test-class path; runtime deps included `pytest-cov`.
3. **`src/` layout was a mess.** Rust code and the Python package shared
   `src/`; a `traits/` layer abstracted a single implementation; `utils/`
   collected unrelated helpers; voxel logic was split across three modules;
   the shipped "random" voxel strategy actually returned the middle index.
4. **Burn moved from compile-time backend selection to a unified runtime
   backend.** Issue #4415 (filed by the project owner) was resolved in burn
   0.21: compile several backends, select per tensor via a `Device` value.
   The repo pinned burn 0.20.1 with a hand-rolled `Router<(Wgpu, NdArray)>`
   alias and a custom auto-detect helper.
5. **The Python dtype contract was ad-hoc.** `from_xyz` accepted 4 dtypes but
   every attribute setter demanded `float32`; RGB was f32 internally / u8
   externally; LAS classification, GPS time, and intensity all lost their
   native types. The owner wants the same dtype promise laspy gives.

Additionally requested on 2026-08-22: **torch-tensor compatibility** for
Python users.

## 2. Findings

### 2.1 Burn 0.21 unified dispatch (source-verified)

- `burn-dispatch` 0.21.0 provides `Dispatch` (a zero-sized backend that
  dispatches per-tensor) and `DispatchDevice` (an enum with one variant per
  compiled backend: `Cpu`, `Cuda`, `Metal`, `Rocm`, `Vulkan`, `Wgpu`, `Flex`,
  `NdArray`, `LibTorch`, `Autodiff`).
- The burn 0.21 facade re-exports it (`pub use burn_dispatch::*`) behind the
  `dispatch` feature. WGPU-flavored features (`metal`/`vulkan`/`webgpu`) are
  mutually exclusive.
- `DispatchDevice::default()` prefers GPU variants, falling back to CPU
  variants — exactly the runtime-selection behavior #4415 asked for.
- The `tch` feature adds a `LibTorch` device: burn tensors *are* libtorch
  tensors on that device.
- `TensorData { bytes, shape, dtype }` carries any of
  F64/F32/F16/BF16/I64/I32/I16/I8/U64/U32/U16/U8/Bool; `float_cast` supports
  runtime float-dtype changes.

### 2.2 laspy dtype contract (doc/source-verified)

- Point records are structured numpy arrays; every standard dimension has a
  pinned dtype (`intensity` u2, `return_number`/`classification` u1,
  `gps_time` f8, `red`/`green`/`blue`/`nir` u2, flags bool, …).
- Raw `X/Y/Z` are i4; user-facing `x/y/z` are f8 computed via header
  `scales`/`offsets`.
- Extra dimensions declare a dtype from {u1..u8, i1..i8, f4, f8} and
  round-trip exactly.
- Setters coerce with numpy assignment semantics and raise on overflow.

### 2.3 Torch compatibility options

| Option | Cost | Verdict |
|---|---|---|
| A. Link libtorch always (tch backend in default build) | Heavy wheels, libtorch version pinning for every user | Rejected as default |
| B. `torch` cargo feature → `DispatchDevice::LibTorch` | Opt-in build; tensors natively torch | Accepted (opt-in) |
| C. Boundary interop via DLPack: accept `torch.Tensor` inputs, export via `torch.from_numpy` | No build cost, zero-copy on CPU | Accepted (default) |

B and C are complementary: C gives every user frictionless torch I/O; B gives
torch-native execution to those who build with it.

### 2.4 Coordinate dtype vs GPU compute

laspy promises `x/y/z` as f8, but wgpu has no f64. Storing f64 tensors would
forbid GPU execution; storing plain f32 breaks the promise (UTM-scale
coordinates lose ~3 cm at f32). Resolution: adopt LAS's own trick — store an
f64 per-cloud `offset` plus f32 *relative* coordinates on the tensor device.
Getters reconstruct f8; translations touch only the offset (exact, f64);
rotations act on small-magnitude relative coords where f32 is ample.

## 3. Decisions (recorded as ADRs)

1. **ADR-0001 — Unified dispatch backend & torch interop.** burn 0.21
   `Dispatch`/`DispatchDevice`, default features CPU + Vulkan, opt-in `cuda`
   and `torch` features, DLPack boundary interop.
2. **ADR-0002 — LAS-aligned dimension & dtype contract.** Standard dimension
   table with laspy dtypes, typed extra dimensions, coercing setters,
   f64-offset + f32-relative coordinate storage.
3. **ADR-0003 — Repository layout & dev workflow.** Rust-only `src/`, Python
   package in `python/`, `docs/rfc` vs `docs/plans` split, deletion of the
   traits layer, single-command test/bench workflow.
4. **ADR-0004 — Multi-format file I/O.** `read`/`write` entry points with
   extension dispatch; LAS/LAZ, Parquet, CSV; dtype-preserving columns;
   polars dropped for direct arrow/parquet.
5. **ADR-0005 — Python API shape.** Single `PointCloud` class; batch-only
   access (no single-point editing); Python wrapper by composition over the
   `_core` extension class.

## 4. Out of scope

- GPU-native voxel binning, selection ops, kNN/normals, outlier removal,
  registration — future RFCs.
- Waveform packet dimensions (LAS formats 4/5/9/10): parked until a user
  needs them; the dimension table reserves their names/dtypes.
- `DownsampleStrategy.AVERAGE` (synthetic centroid points): deferred — it
  requires per-dtype reduction semantics (mean for floats, mode for ints).

## 5. Open questions

- Should `to_device` move typed attribute columns too, once selection ops
  land on GPU? Today attributes live CPU-side; only coordinates ride the
  tensor device.
- When burn stabilizes `Flex` as the default CPU backend, revisit the
  `cpu` (CubeCL) choice in ADR-0001.

---

## Finding appended 2026-08-22 (post-implementation)

The CubeCL CPU backend (`burn/cpu`) segfaults at runtime on large matmuls:
a `[10M,3] x [3,3]` f32 matmul on `DispatchDevice::Cpu` crashes with SIGSEGV
after ~300s of runaway JIT work, while `[5M,3]` completes in 2.4s. The `cpu`
feature therefore maps to `burn/flex` (see ADR-0001). This resolves the §5
open question in flex's favor early; worth reporting upstream to burn with
the minimal reproduction.
