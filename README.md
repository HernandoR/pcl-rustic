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

### Compared to Open3D and laspy

`just bench-compare` measures the same operations against Open3D 0.19 and
laspy 2.7 on the same host. Current standing, stated plainly:

| Operation | Points | pcl-rustic | baseline | Result |
|---|---|---|---|---|
| `voxel_downsample` (0.15) | 1M | 0.91 s | Open3D 0.66 s | 1.4x slower |
| `voxel_downsample` (0.15) | 10M | 11.3 s | Open3D 11.7 s | ~parity |
| `transform` (4x4, warm) | 1M | 12 ms | Open3D 3 ms | 4x slower |
| `transform` (4x4, warm) | 10M | 481 ms | Open3D 29 ms | 17x slower |
| LAS read + xyz | 2M | 0.25 s | laspy 0.04 s | 6.8x slower |

**pcl-rustic is not currently faster than Open3D on CPU.** Two structural
reasons, both consequences of ADR-0002 rather than incidental inefficiency:

- Coordinates are stored as f32 relative to an f64 offset, so every `.xyz`
  read materializes and converts a fresh (N, 3) f8 array. At 10M points that
  readback is 362 ms of the 481 ms transform. Open3D stores f64 natively and
  returns a zero-copy view.
- Open3D's `voxel_down_sample` averages each voxel; our
  `NEAREST_TO_CENTROID` additionally finds the nearest existing point.

What pcl-rustic does deliver against these baselines: exact dtype parity with
laspy (`intensity` u2, `classification` u1, verified on a real LAS file), GPU
eligibility for the coordinate plane, and LAS/LAZ I/O that Open3D lacks
entirely. Coordinate values are not bit-identical to laspy -- the f32
relative storage deviates by up to 3.0e-5 m at a +/-500 m extent, which is 3%
of the default 1 mm LAS scale step and well inside file quantization.

Closing the CPU gap is not yet scheduled work; the readback path is the
first place to look.

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
