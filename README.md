# pcl-rustic

[![CI](https://github.com/ArkWhale/pcl-rustic/workflows/CI/badge.svg)](https://github.com/ArkWhale/pcl-rustic/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/pcl-rustic?label=PyPI)](https://pypi.org/project/pcl-rustic/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.89+-orange)](https://www.rust-lang.org/)
![License](https://img.shields.io/badge/license-MIT-green)

**pcl-rustic** is a point cloud library built in Rust with a PyO3 binding,
using [burn](https://github.com/tracel-ai/burn) tensors for batch compute.
It targets one goal: let a [laspy](https://laspy.readthedocs.io/) user move
to pcl-rustic without changing a single dtype.

## Aims

- **The laspy dtype promise.** Every standard point dimension (`x`,
  `intensity`, `classification`, `gps_time`, `red`, ...) has a pinned numpy
  dtype identical to laspy's, so LAS/LAZ files, CSV exports, and Parquet
  tables all round-trip losslessly. See the [dtype contract](#dtype-contract)
  below and `docs/plans/adr-0002-las-aligned-dimension-dtype-contract-2026-08-22.md`.
- **One backend, any device.** burn 0.21's unified dispatch backend
  ([tracel-ai/burn#4415](https://github.com/tracel-ai/burn/issues/4415))
  lets a single compiled wheel run on CPU or GPU, picking the backend at
  runtime via a device string instead of a compile-time feature.
- **Torch interop.** Every array-accepting API also accepts a torch tensor
  (zero-copy on CPU via DLPack); point clouds export back to torch with
  `to_torch()`. An opt-in `torch` Cargo feature runs burn ops natively on
  libtorch, including its CUDA allocator.
- **Multi-format I/O.** One `PointCloud.read` / `pc.write` pair dispatches
  on file extension across LAS, LAZ, CSV, and Parquet, preserving column
  dtypes end-to-end.

## Install

```bash
uv add pcl-rustic
# or
pip install pcl-rustic
```

### From source

Requires Python 3.10+ and Rust 1.89+ (burn 0.21 uses edition 2024):

```bash
git clone https://github.com/ArkWhale/pcl-rustic.git
cd pcl-rustic
just install
just dev
```

## Quickstart

### NumPy

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

xyz = np.random.default_rng(0).normal(size=(100_000, 3)) * 10.0
pc = PointCloud.from_xyz(xyz)  # any real dtype in, f8 out

pc.intensity = np.random.default_rng(1).integers(0, 65535, size=len(pc))
pc.classification = np.zeros(len(pc), dtype=np.uint8)

pc_down = pc.voxel_downsample(0.15, strategy=DownsampleStrategy.NEAREST_TO_CENTROID)
print(f"{len(pc):,} -> {len(pc_down):,} points")
```

### laspy-style dimension access

```python
pc = PointCloud.read("scan.laz")

# dict-style
pc["classification"]
pc["classification"] = new_classes

# attribute sugar for standard and already-declared extra dimensions
pc.classification
pc.gps_time

# extra dimensions must be declared once before use
pc.add_extra_dim("confidence", "f4")
pc.confidence = confidence_scores
```

### Torch

```python
import torch
from pcl_rustic import PointCloud

xyz = torch.rand((10_000, 3), dtype=torch.float64) * 100.0
pc = PointCloud.from_xyz(xyz)          # zero-copy ingest via DLPack (CPU)
pc["intensity"] = torch.randint(0, 65535, (len(pc),))

intensity_tensor = pc.to_torch("intensity")
all_tensors = pc.to_torch()            # dict[str, torch.Tensor]
```

torch is an optional import: pcl-rustic works with plain numpy if torch is
not installed, and only `to_torch()` / torch-tensor arguments need it.

## Device selection

```python
from pcl_rustic import available_devices, default_device

available_devices()  # e.g. ["cpu", "vulkan"] on Linux, ["cpu", "metal"] on macOS
default_device()     # picks a GPU variant if one initializes, else "cpu"
device_report()      # why each device is / is not usable -- see below

pc.device                  # -> str
pc_gpu = pc.to_device("vulkan")
```

Every wheel carries the CPU backend plus its platform's GPU backend, so the
device is chosen at runtime, never at build time:

| device | availability |
|---|---|
| `cpu` | always (pure-Rust `flex` backend) |
| `metal` | macOS wheels |
| `vulkan` | Linux and Windows wheels |
| `cuda` | opt-in `cuda` Cargo feature |
| `torch` | opt-in `torch` Cargo feature |

macOS gets Metal rather than Vulkan because it has no native Vulkan driver.
The two are mutually exclusive inside burn, so the choice is pinned per target
in `Cargo.toml` instead of being a Cargo feature — see
`docs/plans/adr-0001-unified-dispatch-backend-torch-interop-2026-08-22.md`.
Asking for a device from another platform raises `ValueError` ("not compiled
into this build"); asking for one whose adapter is missing raises `OSError`.

### Linux needs a Vulkan loader

GPU selection degrades to `cpu` rather than raising (ADR-0001), so a host that
*has* a GPU but cannot initialize it looks exactly like a host that has none.
On Linux the usual cause is a missing Vulkan **loader**: the NVIDIA driver
installs an ICD at `/etc/vulkan/icd.d/nvidia_icd.json`, but wgpu talks to
`libvulkan.so.1`, which ships separately and is absent from most server and
container images. Without it wgpu enumerates zero adapters.

```console
$ sudo apt install libvulkan1      # Debian/Ubuntu
$ sudo dnf install vulkan-loader   # Fedora/RHEL
```

Use `device_report()` to tell the two cases apart — it surfaces the underlying
adapter-enumeration error instead of just omitting the device:

```python
>>> from pcl_rustic import device_report
>>> device_report()
{'cpu': 'available',
 'metal': 'not compiled into this build',
 'vulkan': 'unavailable: panicked at ...: No possible adapter available for '
           'backend. ... active_backends: Backends(0x0) ...',
 ...}
```

`active_backends: Backends(0x0)` with `supported_backends` non-empty is the
missing-loader signature. Probing is not cached, so call it when diagnosing,
not on a hot path.

## Coordinate dtype

Coordinates have a process-wide default dtype, torch-style:

```python
import pcl_rustic

pcl_rustic.get_default_dtype()          # "float32" (the default)
pcl_rustic.set_default_dtype("float64") # affects clouds constructed afterwards
pc.dtype                                # np.dtype, fixed at construction
```

or set the initial value with the `PCL_RUSTIC_DEFAULT_DTYPE` environment
variable (`float32`/`float64`; read once, at first use; an invalid value
warns and falls back to `float32` rather than failing the import).

| dtype | storage | reads return | placement at construction |
|---|---|---|---|
| `float32` (default) | f32 relative to a per-cloud f64 offset, on CPU and GPU alike | `float32` | `default_device()` (GPU-first) |
| `float64` | absolute f64 on CPU; GPU is still f32-relative | `float64` | **always CPU** |

`float32` makes CPU and GPU behave identically -- a cloud moves between them
without re-encoding or precision change -- and coordinate reads skip the
f32 -> f64 widening pass, which dominated GPU materialization (measured at
10M points: `.xyz` readback 252 ms -> 125 ms on an A10G, 161 ms -> 17 ms on
CPU). Internal storage is *relative*, so transforms and LAS writes keep f64
anchoring regardless; only the absolute values handed to Python quantize to
the f32 grid. That grid is extent-dependent: ~1e-5 m for a cloud spanning
hundreds of meters, but **~0.25 m at raw UTM magnitudes (4e6)** -- surveying
workflows at absolute coordinates should opt into `float64`.

`float64` clouds ingest CPU-resident even when a GPU is the default device:
no accelerator representation can hold f64, so honoring GPU-first placement
would round the cloud at construction and silently defeat the opt-in.
Rounding under `float64` only ever happens through an explicit
`to_device(...)`. On CPU the f64 representation is absolute and exact: LAS
round-trips are bit-identical to laspy.

## Dtype contract

`PointCloud` is a set of named, equal-length dimensions. Standard dimension
names carry pinned numpy dtypes, identical to laspy:

| Dimension | numpy dtype | Notes |
|---|---|---|
| `x`, `y`, `z` | `f4` / `f8` | follows the coordinate dtype (below); writes always accepted as `f8` |
| `intensity` | `u2` | |
| `return_number` | `u1` | unpacked from the LAS bit field |
| `number_of_returns` | `u1` | unpacked |
| `synthetic`, `key_point`, `withheld`, `overlap` | `bool` | unpacked flags |
| `scan_direction_flag`, `edge_of_flight_line` | `bool` | unpacked |
| `scanner_channel` | `u1` | LAS formats 6-10 |
| `classification` | `u1` | |
| `user_data` | `u1` | |
| `scan_angle` | `i2` | LAS 0-5 `scan_angle_rank` (i1) widens losslessly |
| `point_source_id` | `u2` | |
| `gps_time` | `f8` | |
| `red`, `green`, `blue` | `u2` | 16-bit color, per LAS -- **not** u8 |
| `nir` | `u2` | |

Getters always return this pinned dtype. Setters accept the exact dtype
as-is, or coerce a same-kind array (int widening/narrowing, float
widening/narrowing) with range validation -- `OverflowError` on
out-of-range values, `TypeError` on a kind-incompatible cast (e.g. a float
array into `classification`). `add_extra_dim(name, dtype)` accepts any of
`u1 u2 u4 u8 i1 i2 i4 i8 f4 f8` and round-trips exactly through Parquet.
LAS write currently skips extra dimensions -- the `las` crate exposes no
typed ExtraBytes API (see ADR-0004); use Parquet as the lossless carrier.

Full rationale: `docs/plans/adr-0002-las-aligned-dimension-dtype-contract-2026-08-22.md`.

## Development

This project uses [just](https://github.com/casey/just) as its command
runner and [uv](https://github.com/astral-sh/uv) for Python packaging.

```bash
just install     # uv sync --all-groups + pre-commit hooks
just dev         # maturin develop (debug build)
just build       # maturin develop --release
just test        # fast suite: pytest -m "not slow and not bench"
just test-all    # everything except benchmarks
just bench       # throughput benchmarks: pytest -m bench -s
just bench-compare  # same, plus Open3D/laspy comparison (installs `bench` group)
just test-rust   # cargo test --release
just fmt         # cargo fmt + ruff format
just lint        # cargo clippy + ruff check
just ci          # pre-commit + rust tests + fast suite
```

Reference throughput on a 96-thread AMD EPYC 7R13 host (CPU `flex` device,
release build, `just bench`):

| Operation | Points | Time | Throughput |
|---|---|---|---|
| `voxel_downsample` (0.15) | 1M | 1.2 s | 0.83M pts/s |
| `voxel_downsample` (0.15) | 10M | 12.2 s | 0.82M pts/s |
| `transform` (4x4, incl. readback) | 1M | 0.07 s | 15.0M pts/s |
| `transform` (4x4, incl. readback) | 10M | 0.51 s | 19.6M pts/s |

Numbers vary by host; run `just bench` locally.

### Compared to Open3D, PCL, and laspy

`just bench-compare` measures against Open3D 0.19 and laspy 2.7 (both in the
`test` dependency group); `just bench-pcl` additionally builds a small C++
harness (`bench/pcl_cpp/`) and measures against PCL 1.14. Every baseline is
optional and skips when absent.

Methodology: one warmup call, then the median of three timed runs, applied
identically to every engine including the C++ harness. Without the warmup,
pcl-rustic's first op in a process pays a one-time kernel-init cost (~55 ms)
and results depend on test ordering.

Current standing on a 96-thread EPYC 7R13, after the RFC-0001 optimisation
pass:

| Operation | Points | pcl-rustic | baseline | Result |
|---|---|---|---|---|
| `voxel_downsample` (0.15) | 1M | 0.12 s | Open3D 0.56 s | **4.8x faster** |
| `voxel_downsample` (0.15) | 10M | 1.63 s | Open3D 9.93 s | **6.1x faster** |
| `voxel_downsample` (leaf 1.0) | 2M | 0.28 s | PCL C++ 0.17 s | 1.6x slower |
| `transform` (4x4) | 1M | 9 ms | Open3D 3 ms | 3.2x slower |
| `transform` (4x4) | 10M | 50 ms | Open3D 28 ms | 1.8x slower |
| `transform` (4x4) | 2M | 14 ms | PCL C++ 4 ms | 3.5x slower |
| LAS read + xyz | 2M | 0.07 s | laspy 0.03 s | 2.1x slower |

Coordinates are **bit-identical to laspy** on a LAS round-trip (measured
deviation exactly 0.0), because CPU-resident clouds hold absolute f64.

For reference, the same table before the optimisation pass read 1.8x *slower*
than Open3D on 1M voxel downsampling, 10.5x slower than PCL, and carried a
3.03e-5 m coordinate deviation. What changed:

- **Voxel downsampling** replaced a `HashMap<[i64;3], Vec<u32>>` -- one heap
  allocation per voxel -- with sort-based grouping: sort point indices by
  voxel key once, then scan runs. Cost no longer scales with voxel count.
- **Coordinates** are stored as absolute f64 on CPU instead of f32 relative
  to an offset, which removed the f32->f64 widening pass from every read and
  closed the laspy fidelity gap (ADR-0002, amended).
- **Transforms** on CPU use a specialised rayon `R * p + t` kernel instead of
  a general tensor matmul. Measured on this host, a plain loop does 10M
  points in 23 ms where burn's matmul needs 173-273 ms; the degenerate
  `[N,3] x [3,3]` shape wins nothing from a tiled kernel. (A 4x4 homogeneous
  matmul is faster *inside* burn -- 173 ms vs 273 ms, since 4-wide suits the
  kernel -- but only if points are stored pre-padded as PCL does; padding
  packed (N,3) data per call costs more than it saves.)
- **LAS reading** bulk-decodes the uncompressed point block in parallel
  instead of materialising a `Point` struct per record. LAZ is unchanged,
  since LASzip records have no fixed byte layout to slice.

What remains: the `.xyz` readback still copies (143 ms at 10M points), which
is now the dominant term in a transform. Open3D pays nothing there because it
returns a view into its own f64 buffer. Doing the same safely needs
copy-on-write coordinate storage so an outstanding view cannot dangle when a
setter replaces the buffer; that is deferred, not done.

## Documentation layout

- `docs/rfc/` -- append-only RFC discussion logs (design exploration, kept
  even after resolution).
- `docs/plans/` -- Architecture Decision Records (ADRs), the current,
  atomically maintained source of truth for accepted decisions.

Start with `docs/rfc/rfc-0001-full-structure-redesign-2026-08-22.md` for the
rationale behind this redesign, and `docs/plans/index.md` for the full ADR
index.

## License

MIT License -- see [LICENSE](LICENSE).
