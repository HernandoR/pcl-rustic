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

---

## Finding appended 2026-08-23 (upstream triage)

Standalone reproductions were built for the three upstream defects worked
around during implementation, and each was checked against existing upstream
reports:

1. **burn `tch` feature does not reach `burn-dispatch`** — *not* designed
   behaviour, and already fixed upstream. In burn 0.21.0 every backend
   feature forwards (`cpu`, `cuda`, `flex`, `metal`, `ndarray`, `rocm`,
   `vulkan`, `webgpu`, `wgpu` all list `burn-dispatch?/X`) except
   `tch = ["burn-tch", "burn-vision?/tch"]`, while `burn-dispatch/tch`
   exists and gates the documented `DispatchDevice::LibTorch` variant. In
   0.22 the chain `burn/tch` → `burn-core/tch` → `burn-tensor/tch` =
   `burn-dispatch/tch` closes the gap. No issue worth filing; our phantom
   `burn-dispatch` dependency is dropped on the 0.22 upgrade.

2. **CubeCL CPU backend aborts on tall matmuls** — no existing upstream
   issue found. Refined diagnosis: it is a *stack overflow on an internal
   cubecl worker thread*, not a plain SIGSEGV, so callers cannot work around
   it by enlarging their own stack. A `[N,3] × [3,3]` f32 matmul succeeds at
   N = 5,000,000 (2.3 s) and aborts at N = 5,592,405 and above, after ~170 s
   of runaway work. Reproduces identically on burn 0.21.0 and 0.22.0-pre.2;
   `flex` runs N = 10,000,000 in 0.22 s. Filed as tracel-ai/burn#5419.

3. **laz GPS-time subtraction overflow** — no open issue, but strong
   precedent: laz-rs #13 fixed the same class of panic in `rgb.rs`, and #33
   records the maintainer's policy that these operations are *intended* to
   wrap and that debug-build overflow panics are found and fixed by
   converting them to `wrapping_*`. In `laz-0.12.2/src/las/gps.rs` line 556
   already uses `this_val.value.wrapping_sub(...)`, while line 581 performs
   the same subtraction with a raw `-`. One-line fix, filed as laz-rs/laz-rs#72
   with a pure-`laz` reproducer (500 points, panics in debug, clean in
   release).

---

## Finding appended 2026-08-23 (baseline comparison)

`tests/test_benchmark_compare.py` (`just bench-compare`) now measures against
Open3D 0.19 and laspy 2.7. PCL itself is not benchmarked: its Python bindings
are dead (`pclpy` requires `==3.6.*` with a Windows-only wheel, `python-pcl`'s
newest release is a 2018 cp27/cp35 alpha), so a PCL comparison would require a
separate C++ harness.

Result on a 96-thread EPYC 7R13, CPU `flex` device:

| Operation | Points | pcl-rustic | baseline | Result |
|---|---|---|---|---|
| `voxel_downsample` | 1M | 0.92 s | Open3D 0.66 s | 1.4x slower |
| `voxel_downsample` | 10M | 12.1 s | Open3D 12.3 s | parity |
| `transform` (warm) | 1M | 12 ms | Open3D 3 ms | 4x slower |
| `transform` (warm) | 10M | 481 ms | Open3D 29 ms | 17x slower |
| LAS read + xyz | 2M | 0.24 s | laspy 0.04 s | 6.5x slower |

**We are not faster than Open3D on CPU today.** The dominant cost is
structural, not incidental: ADR-0002 stores coordinates as f32 relative to an
f64 offset, so every `.xyz` read materializes and converts a fresh (N, 3) f8
array. Instrumented at 10M points, a warm transform splits into 153 ms of
submit and **362 ms of readback**. Open3D keeps f64 natively and hands back a
zero-copy view, so it pays neither cost.

What the comparison does confirm:

- **Dtype parity with laspy is exact** on a real LAS file (`intensity` u2,
  `classification` u1, values identical) -- the headline goal of the redesign.
- **Coordinate values are not bit-identical to laspy.** The f32 relative
  storage deviates by up to 3.03e-5 m at a +/-500 m extent. That is 3.0% of
  the default 1 mm LAS scale step, so it stays inside file quantization and
  the ADR-0002 claim ("f32 error stays below LAS quantization for typical
  extents") holds -- but "lossless round-trip" is accurate for dimensions and
  *not* for coordinates, and the README was corrected accordingly.

Open questions this raises for a future RFC:

- Should CPU-resident clouds store f64 coordinates directly, keeping the
  f32-relative representation only for GPU devices? That would erase the
  readback conversion and the fidelity gap at once, at the cost of two
  coordinate representations.
- Should `.xyz` cache its materialized array and invalidate on mutation?
  Repeated reads currently re-convert every time.

---

## Finding appended 2026-08-23 (PCL C++ baseline)

PCL is now benchmarked too, via a C++ harness (`bench/pcl_cpp/`, `just
bench-pcl`) rather than a Python binding, since none exists for 3.12. The
harness reads the same f64 xyz dump the Python benchmarks write, so all three
engines see byte-identical input.

PCL 1.14, 2M points, 96-thread EPYC 7R13:

| Operation | pcl-rustic | PCL C++ | Result |
|---|---|---|---|
| `voxel_downsample` (leaf 1.0) | 2.21 s | 0.19 s | 11.9x slower |
| `transform` (4x4) | 0.14 s | 0.02 s | 7.6x slower |

Same conclusion as the Open3D comparison, more starkly: our CPU compute is
not competitive. PCL uses f32 end to end with no Python boundary and no
readback conversion.

Two things the PCL comparison established that the Open3D one did not:

- **Output parity.** At a leaf size PCL accepts, PCL's `VoxelGrid` and our
  `NEAREST_TO_CENTROID` return the *same* point count (1,955,826 of
  2,000,000), which is a useful correctness cross-check on our voxel
  binning, independent of the timing.
- **PCL has a voxel-count ceiling we do not.** `VoxelGrid` indexes voxels
  with a 32-bit integer and silently returns the input *unfiltered* when the
  grid would overflow it. Measured: leaf 0.15 over a +/-500 m extent gives
  back all 2,000,000 points untouched. `tests/test_benchmark_pcl.py`
  asserts this refusal explicitly so the no-op can never be scored as a fast
  run, and so the behavioural difference stays documented.

This strengthens the case for the ADR-0002 follow-up already noted above: if
CPU-resident clouds held f64 coordinates directly, both the readback cost and
the laspy fidelity gap would disappear together.

---

## Finding appended 2026-08-23 (comparison consolidated into Python)

At the owner's direction the comparison now lives in one Python module,
`tests/test_benchmark_compare.py`, driven by the `test` dependency group
(`uv add open3d --group test`). The separate PCL test module was folded in.

`uv add python-pcl --group test` was attempted and **cannot work on Python
3.12.** Three successive real failures, each further into the build:

1. `ModuleNotFoundError: No module named 'Cython'` (undeclared build dep);
2. after adding Cython, `No module named 'numpy'` (same);
3. after adding both, `NameError: name 'vtk_include_dir' is not defined`.

Failure 3 is terminal: `python-pcl`'s `setup.py` selects VTK paths from
hard-coded `pcl_version` branches for `-1.7`, `-1.8`, and `-1.9`, and falls
through to `else: pass` for anything else. Ubuntu 24.04 ships PCL 1.14, so no
branch matches and the variable is never bound. There is no environment
override, and even a patched `setup.py` would still be compiling 2018-era
Cython sources against removed CPython 3.12 C-API. `pclpy` is likewise
unusable (`requires_python == 3.6.*`, Windows-only wheel). PCL therefore stays
a C++ harness invoked from the Python test, which keeps the whole comparison
runnable and reported from Python.

Two measurement defects were found and fixed while consolidating, both of
which had been producing misleading numbers:

- **Order-dependent timings.** pcl-rustic's first op in a process pays a
  one-time kernel-init cost of roughly 55 ms, so whichever comparison ran
  first was penalised. The earlier "17x slower on a 10M transform" figure was
  this artifact. With one warmup plus the median of three runs -- now applied
  uniformly, including inside the C++ harness -- the 10M transform is 4.0x
  slower, matching the 1M result (4.1x) instead of contradicting it.
- **In-place mutation under repetition.** Open3D's `transform` mutates the
  cloud, so warmup-plus-repeat applied the matrix four times cumulatively and
  broke the numerical cross-check. Timing is unaffected (identical work per
  call), so the fix was to keep the repeated timing and move the correctness
  assertion onto freshly built clouds.

Corrected standing (96-thread EPYC 7R13, warmup + median of 3):

| Operation | Points | pcl-rustic | baseline | Result |
|---|---|---|---|---|
| `voxel_downsample` (0.15) | 1M | 1.11 s | Open3D 0.63 s | 1.8x slower |
| `voxel_downsample` (0.15) | 10M | 11.96 s | Open3D 9.31 s | 1.3x slower |
| `voxel_downsample` (leaf 1.0) | 2M | 1.82 s | PCL 0.17 s | 10.5x slower |
| `transform` | 1M | 12 ms | Open3D 3 ms | 4.1x slower |
| `transform` | 10M | 112 ms | Open3D 28 ms | 4.0x slower |
| `transform` | 2M | 24 ms | PCL 4 ms | 5.7x slower |
| LAS read + xyz | 2M | 0.25 s | laspy 0.04 s | 7.1x slower |

The readback conclusion holds and is now quantified as a share rather than a
single figure: the `.xyz` materialization is 63% of a 1M transform and 72% of
a 10M one. It is a large transient allocation, so the absolute number is
allocator-sensitive -- the same 10M transform measures 112 ms to 509 ms
depending on whether the previous result buffer is still live when the next is
allocated. That sensitivity is itself an argument for the ADR-0002 follow-up:
holding f64 coordinates directly for CPU-resident clouds would remove the
allocation, the conversion, and the laspy fidelity gap together.
