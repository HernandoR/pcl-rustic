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

available_devices()  # e.g. ["cpu", "vulkan"], or add "cuda"/"torch" if built with those features
default_device()     # picks a GPU variant if one initializes, else "cpu"

pc.device                  # -> str
pc_gpu = pc.to_device("vulkan")
```

Device names map 1:1 to the backends the wheel was compiled with (`cpu`,
`vulkan` on by default; `cuda` and `torch` are opt-in Cargo features).

## Dtype contract

`PointCloud` is a set of named, equal-length dimensions. Standard dimension
names carry pinned numpy dtypes, identical to laspy:

| Dimension | numpy dtype | Notes |
|---|---|---|
| `x`, `y`, `z` | `f8` | scaled coordinates |
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

Current standing on a 96-thread EPYC 7R13:

| Operation | Points | pcl-rustic | baseline | Result |
|---|---|---|---|---|
| `voxel_downsample` (0.15) | 1M | 1.11 s | Open3D 0.63 s | 1.8x slower |
| `voxel_downsample` (0.15) | 10M | 11.96 s | Open3D 9.31 s | 1.3x slower |
| `voxel_downsample` (leaf 1.0) | 2M | 1.82 s | PCL C++ 0.17 s | 10.5x slower |
| `transform` (4x4) | 1M | 12 ms | Open3D 3 ms | 4.1x slower |
| `transform` (4x4) | 10M | 112 ms | Open3D 28 ms | 4.0x slower |
| `transform` (4x4) | 2M | 24 ms | PCL C++ 4 ms | 5.7x slower |
| LAS read + xyz | 2M | 0.25 s | laspy 0.04 s | 7.1x slower |

**pcl-rustic is not currently faster than Open3D or PCL on CPU.** Two
structural reasons, both consequences of ADR-0002 rather than incidental
inefficiency:

- Coordinates are stored as f32 relative to an f64 offset, so every `.xyz`
  read materializes and converts a fresh (N, 3) f8 array. Instrumented, that
  readback is 63% of a 1M transform and 72% of a 10M one. It is also a large
  transient allocation, which makes the timing allocator-sensitive: the same
  10M transform measures anywhere from 112 ms to 509 ms depending on whether
  the previous result buffer is still alive when the next is allocated.
  Open3D stores f64 natively and returns a zero-copy view; PCL uses f32
  throughout with no Python boundary.
- Open3D's `voxel_down_sample` and PCL's `VoxelGrid` both average each voxel;
  our `NEAREST_TO_CENTROID` additionally finds the nearest existing point.

What pcl-rustic does deliver against these baselines: exact dtype parity with
laspy (`intensity` u2, `classification` u1, verified on a real LAS file), GPU
eligibility for the coordinate plane, LAS/LAZ I/O that Open3D lacks entirely,
and a voxel grid without PCL's size limit -- PCL's `VoxelGrid` indexes voxels
with a 32-bit integer and *silently returns the cloud unfiltered* when the
grid overflows it (measured: leaf 0.15 over a +/-500 m extent returns all
2,000,000 points untouched, where pcl-rustic downsamples normally). At a leaf
size PCL accepts, the two agree exactly on output size (1,955,826 points).

Coordinate values are not bit-identical to laspy: the f32 relative storage
deviates by up to 3.0e-5 m at a +/-500 m extent, which is 3% of the default
1 mm LAS scale step and well inside file quantization.

Closing the gap is not yet scheduled work, but it has been profiled rather
than guessed (see the profiling section of RFC-0001). In short: the `.xyz`
readback is already within 12% of what numpy needs for the same f32->f64
conversion, so it is the *materialization itself* that must be designed away
by holding f64 coordinates for CPU-resident clouds; voxel downsampling costs
~855 ns per voxel because it allocates a `Vec` per voxel instead of sorting
once like PCL; and LAS reading spends ~139 ns per point building a `Point`
struct where laspy memcpys the record block.

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
