# pcl-rustic

High-performance Python point-cloud library built on Rust, PyO3, and the
[Burn](https://github.com/tracel-ai/burn) tensor framework.

## Aims

- **laspy-compatible dtype promise** — every standard dimension carries the
  numpy dtype laspy users expect (`intensity` u2, `classification` u1,
  `gps_time` f8, `red/green/blue` u2, `x/y/z` f8). See
  [ADR-0002](plans/adr-0002-las-aligned-dimension-dtype-contract-2026-08-22.md).
- **Unified runtime backend** — one wheel runs on CPU and GPU via burn's
  dispatch backend
  ([tracel-ai/burn#4415](https://github.com/tracel-ai/burn/issues/4415)). See
  [ADR-0001](plans/adr-0001-unified-dispatch-backend-torch-interop-2026-08-22.md).
- **Torch interop** — `torch.Tensor` accepted everywhere arrays are, plus an
  optional LibTorch execution backend.
- **Multi-format I/O** — LAS/LAZ, Parquet, and CSV with dtype-preserving
  round-trips.

## Quickstart

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

pc = PointCloud.from_xyz(np.random.randn(10_000, 3) * 100)
pc["intensity"] = np.random.randint(0, 65535, len(pc), dtype=np.uint16)

down = pc.voxel_downsample(0.15, DownsampleStrategy.RANDOM, seed=7)
down.write("down.laz")
```

See the [README](https://github.com/ArkWhale/pcl-rustic#readme) for the full
guide.

## Design records

- [RFC index](rfc/index.md) — append-only design discussions.
- [ADR index](plans/index.md) — current, atomically-maintained design intent.

## License

MIT.
