# RFC-0002: Enhanced I/O Subsystem

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author(s)** | liuzhen19 |
| **Created** | 2026-05-03 |
| **Updated** | 2026-05-03 |
| **Related RFCs** | [RFC-0001](RFC-0001-core-architecture.md) · [RFC-0007](RFC-0007-testing-infrastructure.md) |

## Summary

Extend PCL Rustic's I/O subsystem to support **streaming reads/writes for large files**, **automatic format detection**, a unified `from_file` / `to_file` API, and a fully-tested Parquet round-trip. Introduce a chunked reader interface so that files exceeding RAM can be processed incrementally.

## Motivation

Current limitations of the I/O layer (RFC-0001):

1. **No streaming**: `from_las` and `from_parquet` load the entire file into RAM before exposing any points, which fails for files > available RAM.
2. **No auto-detection**: callers must know the file format and call the right method.
3. **Parquet implementation is stub-level**: `from_parquet` / `to_parquet` exist but lack round-trip tests and have no support for attribute columns beyond XYZ.
4. **No batch/pipeline support**: to process a 50 GB LAS file you must split it beforehand.

### Use Cases

- Tile-based LiDAR processing pipelines (city-scale scans, 100 GB+).
- Conversion utilities that need to detect and re-encode arbitrary formats.
- Memory-constrained environments (edge devices, containers with 2 GB RAM limits).

## Design

### 3.1 Unified `from_file` / `to_file` API

Add format auto-detection based on file extension and magic bytes:

```python
# Python API (new)
pc = PointCloud.from_file("scan.las")       # auto-detects LAS
pc = PointCloud.from_file("scan.laz")       # auto-detects LAZ
pc = PointCloud.from_file("data.parquet")   # auto-detects Parquet
pc = PointCloud.from_file("data.csv")       # auto-detects CSV

pc.to_file("output.las")
pc.to_file("output.parquet")
```

Rust implementation:

```rust
pub fn from_file(path: &str) -> Result<HighPerformancePointCloud> {
    match detect_format(path)? {
        FileFormat::Las | FileFormat::Laz => from_las(path),
        FileFormat::Parquet               => from_parquet(path),
        FileFormat::Csv                   => from_csv(path, b','),
        FileFormat::Unknown(ext)          => Err(PointCloudError::UnsupportedFormat(ext)),
    }
}
```

`detect_format` inspects the extension first; if ambiguous, reads the first 8 bytes for magic numbers.

### 3.2 Streaming / Chunked Reader

Introduce a `PointCloudReader` iterator that yields fixed-size `PointCloud` chunks:

```python
# Python API (new)
with PointCloud.open("huge_scan.las", chunk_size=1_000_000) as reader:
    for chunk in reader:              # chunk: PointCloud with ≤1M points
        result = process(chunk)
        result.to_file(f"out_{i}.parquet")
```

Rust side:

```rust
pub struct PointCloudChunkReader {
    inner: Box<dyn ChunkedReader>,
    chunk_size: usize,
}

pub trait ChunkedReader: Send {
    fn next_chunk(&mut self) -> Result<Option<HighPerformancePointCloud>>;
    fn total_points(&self) -> Option<u64>;
}
```

Implementations:
- `LasChunkReader` — wraps `las::Reader` with point-count tracking
- `ParquetChunkReader` — wraps `parquet::arrow::async_reader` row-group iteration
- `CsvChunkReader` — buffered line reader

### 3.3 Full Parquet Round-Trip

Extend the Parquet schema to persist **all** optional attributes:

| Arrow column | Type | Mapping |
|---|---|---|
| `x`, `y`, `z` | `Float32` | `xyz` |
| `intensity` | `Float32`, nullable | `intensity` |
| `r`, `g`, `b` | `UInt8`, nullable | `rgb` |
| `<attr_name>` | `Float32`, nullable | `attributes[attr_name]` |

Use `arrow2` (or `arrow-rs`) for zero-copy serialisation.

### 3.4 Format Registration (Extension Point)

Allow third-party crates to register new formats without forking PCL Rustic:

```rust
pub trait IoPlugin: Send + Sync {
    fn extensions(&self) -> &[&str];
    fn magic(&self) -> Option<&[u8]>;
    fn read(&self, path: &str) -> Result<HighPerformancePointCloud>;
    fn write(&self, pc: &HighPerformancePointCloud, path: &str) -> Result<()>;
}

pub fn register_io_plugin(plugin: Box<dyn IoPlugin>);
```

### 3.5 Dependency Changes

| Crate | Version | Purpose |
|-------|---------|---------|
| `arrow-rs` / `parquet` | 51.x | Parquet + Arrow columnar I/O |
| `las` | 0.8.x | LAS/LAZ streaming |
| `csv` | 1.3.x | CSV buffered I/O (already present) |

### Alternatives Considered

| Approach | Reason Not Chosen |
|----------|-------------------|
| Expose raw bytes via Python `io.RawIOBase` | Adds complexity; callers should use chunks |
| Async Rust (`tokio`) throughout | PyO3 async support is immature; adds build complexity |
| Python-side format detection | Bypasses Rust's type-safe dispatch |

## Impact

### Breaking Changes

- `from_parquet` / `to_parquet` signature may change to include schema options (keyword-only, so non-breaking in Python).
- `from_las` remains unchanged; `from_file` is additive.

### Performance Implications

- Streaming reduces peak RAM from O(file size) to O(chunk_size × point_bytes).
- Parquet round-trip should achieve >2 GB/s read throughput (Arrow IPC path).

### Testing Plan

- `tests/test_io.py` — pytest integration tests covering:
  - Round-trip for each format: LAS, LAZ, Parquet, CSV.
  - Auto-detection correctness for all supported extensions.
  - Chunk reader yields correct point counts; concatenation equals full load.
  - Attribute preservation across Parquet round-trip.
  - Error handling for unsupported formats, truncated files, wrong dtypes.
- CI: runs via `just test` on every PR (see [RFC-0007](RFC-0007-testing-infrastructure.md)).
- API docs: use context7 to verify `arrow-rs` and `las` crate APIs before implementation.

## Unresolved Questions

1. Should `PointCloudChunkReader` be exposed as a Python context manager or as a generator function?
2. Memory-mapped I/O for very large files — is it worth the added `unsafe` surface?
3. Compression codec selection for Parquet (Snappy vs Zstd)?

## References

- [arrow-rs crate](https://github.com/apache/arrow-rs)
- [las crate docs](https://docs.rs/las)
- [LAS 1.4 specification](https://www.asprs.org/divisions-committees/lidar-division/laser-las-file-format-exchange-activities)
- [Parquet format specification](https://parquet.apache.org/docs/)
- [RFC-0001: Core Architecture](RFC-0001-core-architecture.md)
