# RFC-0003: Extended File Format Support

- **Status**: Draft
- **Author(s)**: liuzhen19 <liuzhen19@xiaomi.com>
- **Created**: 2026-05-03
- **Updated**: 2026-05-03

---

## Summary

Extend pcl-rustic's I/O capabilities to fully support Parquet and add E57, PLY, and PCD formats.  
Parquet support is already scaffolded in the codebase but not exposed in the Python API.  
This RFC formalises the complete `IOConvert` trait surface and specifies per-format column/attribute mappings.

---

## Motivation

The current Python-visible I/O is limited to LAS/LAZ and CSV.  
Parquet is widely used in ML/data pipelines (Spark, Arrow, DuckDB).  
PLY is the most common exchange format for academic point cloud datasets (ShapeNet, ModelNet).  
PCD is the native format for ROS/PCL interoperability.  
E57 is mandated in AEC (architecture, engineering, construction) workflows.

Gaps:
- `src/io/parquet.rs` exists but is not wired into the Python bindings.
- No PLY, PCD, or E57 support.
- No schema specification for custom attribute columns in Parquet.

---

## Detailed Design

### Extended `IOConvert` Trait

```rust
pub trait IOConvert: Sized {
    // Existing
    fn from_las(path: &str) -> Result<Self>;
    fn to_las(&self, path: &str, compress: bool) -> Result<()>;
    fn from_csv(path: &str, delimiter: u8) -> Result<Self>;
    fn to_csv(&self, path: &str, delimiter: u8) -> Result<()>;

    // New
    fn from_parquet(path: &str) -> Result<Self>;
    fn to_parquet(&self, path: &str) -> Result<()>;
    fn from_ply(path: &str) -> Result<Self>;
    fn to_ply(&self, path: &str, binary: bool) -> Result<()>;
    fn from_pcd(path: &str) -> Result<Self>;
    fn to_pcd(&self, path: &str, binary: bool) -> Result<()>;
    fn from_e57(path: &str) -> Result<Self>;      // read-only initially
}
```

### Python API

```python
# Parquet
pc = PointCloud.from_parquet("scan.parquet")
pc.to_parquet("out.parquet")

# PLY
pc = PointCloud.from_ply("mesh.ply")
pc.to_ply("out.ply", binary=True)

# PCD (ROS compatible)
pc = PointCloud.from_pcd("scan.pcd")
pc.to_pcd("out.pcd", binary=False)

# E57 (read-only v1)
pc = PointCloud.from_e57("scan.e57")
```

### Parquet Schema

| Column name | Arrow type | Maps to |
|-------------|-----------|---------|
| `x`, `y`, `z` | `Float32` | `xyz` |
| `intensity` | `Float32` | intensity (optional) |
| `r`, `g`, `b` | `Float32` | RGB channels (optional) |
| any other | `Float32` | custom attribute |

Extra columns that cannot be parsed as `Float32` are silently ignored with a warning.

### PLY Mapping

Standard scalar properties `x y z` → xyz.  
`nx ny nz` → `normals` attribute (if present).  
`red green blue` / `diffuse_red …` → rgb.  
All other scalar properties → named attributes.

### PCD Mapping

`FIELDS x y z` → xyz.  
`FIELDS x y z intensity` → xyz + intensity.  
Binary compressed PCD is supported read-only initially.

### Rust Crate Selection

| Format | Crate | License |
|--------|-------|---------|
| Parquet | `parquet` + `arrow` (already in Cargo.toml) | Apache-2.0 |
| PLY | `ply-rs` | MIT |
| PCD | `pcd-rs` | MIT |
| E57 | `e57` | MIT |

---

## Implementation Plan

- [ ] Wire `src/io/parquet.rs` into `src/lib.rs` Python bindings
- [ ] Add `from_parquet` / `to_parquet` to `HighPerformancePointCloud`
- [ ] Add `src/io/ply.rs` using `ply-rs`
- [ ] Add `src/io/pcd.rs` using `pcd-rs`
- [ ] Add `src/io/e57.rs` (read-only) using `e57` crate
- [ ] Expose all new methods in PyO3 bindings
- [ ] Add `tests/test_io.py` with round-trip tests for each format
- [ ] Update `docs/api/io.md`
- [ ] Add file extension auto-detection helper: `PointCloud.from_file(path: str)`

---

## Testing Strategy

```python
# tests/test_io.py
import pytest, tempfile, numpy as np
from pathlib import Path
from pcl_rustic import PointCloud

@pytest.fixture
def sample_pc():
    xyz = np.random.randn(500, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc.set_intensity(np.random.rand(500).astype(np.float32))
    return pc

def test_parquet_roundtrip(sample_pc, tmp_path):
    p = tmp_path / "test.parquet"
    sample_pc.to_parquet(str(p))
    loaded = PointCloud.from_parquet(str(p))
    assert loaded.point_count() == sample_pc.point_count()
    np.testing.assert_allclose(loaded.get_xyz(), sample_pc.get_xyz(), atol=1e-5)

def test_ply_roundtrip_ascii(sample_pc, tmp_path):
    p = tmp_path / "test.ply"
    sample_pc.to_ply(str(p), binary=False)
    loaded = PointCloud.from_ply(str(p))
    assert loaded.point_count() == sample_pc.point_count()

def test_ply_roundtrip_binary(sample_pc, tmp_path):
    p = tmp_path / "test.ply"
    sample_pc.to_ply(str(p), binary=True)
    loaded = PointCloud.from_ply(str(p))
    assert loaded.point_count() == sample_pc.point_count()

def test_pcd_roundtrip(sample_pc, tmp_path):
    p = tmp_path / "test.pcd"
    sample_pc.to_pcd(str(p), binary=False)
    loaded = PointCloud.from_pcd(str(p))
    assert loaded.point_count() == sample_pc.point_count()

def test_auto_from_file(sample_pc, tmp_path):
    for ext in ["parquet", "ply", "pcd"]:
        p = tmp_path / f"test.{ext}"
        getattr(sample_pc, f"to_{ext}")(str(p))
        loaded = PointCloud.from_file(str(p))
        assert loaded.point_count() == sample_pc.point_count()
```

---

## Alternatives Considered

- **Delegate Parquet/Arrow to Python layer (pandas/pyarrow)**: avoids adding Rust crates but breaks the zero-copy goal and requires an optional heavy Python dependency.
- **Only implement Parquet**: insufficient; PLY/PCD are essential for academic/ROS workflows.

---

## Drawbacks

- Binary size grows with each new crate (~1–3 MB per format).
- `e57` crate is relatively young; may require patching.
- PLY ASCII parsing is slow for large files (acceptable; recommend binary PLY for production).

---

## Unresolved Questions

- Should `to_parquet` support row-group size configuration for large files?
- Should we support multi-scan E57 files (multiple `Data3D` entries)?
- Do we expose a `PointCloud.from_file(path)` that auto-detects format by extension, or keep explicit methods?

---

## Related RFCs

- [RFC-0001](./RFC-0001-foundation.md) — IOConvert trait is defined there; this RFC extends it
