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

---

## Review Notes — Reviewer A

**Reviewer:** Reviewer A  
**Date:** 2026-05-03  
**Status:** Changes Requested

---

### 1. Naming mismatch with existing implementation

The RFC proposes `PointCloud.from_file()` / `pc.to_file()` (§3.1) and `PointCloud.open()` (§3.2), but the **codebase already ships** `PointCloud.load_from_file()` / `pc.save_to_file()` (see `src/lib.rs` lines 471–527 and `src/io/mod.rs`). The RFC must either:

- (a) adopt the existing names and describe this RFC as *extending* the already-landed API, or  
- (b) explicitly propose a rename from `load_from_file` → `from_file` and treat that as a breaking change.

As written, a reader cannot tell whether `from_file` is new or an alias. This ambiguity will cause confusion during implementation review.

---

### 2. Async contradiction in §3.2

Section 3.2 proposes:

> `ParquetChunkReader` — wraps `parquet::arrow::async_reader` row-group iteration

But the *Alternatives Considered* table explicitly rejects:

> **Async Rust (`tokio`) throughout** — PyO3 async support is immature; adds build complexity

`parquet::arrow::async_reader` **requires a `tokio` runtime**. You cannot use it in a synchronous PyO3 context without `block_on`, which reintroduces exactly the complexity the alternatives table rules out. Either:

- Use `parquet::arrow::arrow_reader::ParquetRecordBatchReader` (the sync row-group iterator) instead, or  
- Add an explicit "GIL-releasing sync wrapper around `tokio::runtime::Handle::block_on`" to the design and justify it.

This contradiction must be resolved before implementation begins.

---

### 3. Dependency section is out of date and incomplete

The RFC's dependency table (§3.5) has two problems:

| RFC claims | Cargo.toml reality |
|---|---|
| `arrow-rs` / `parquet` 51.x | `arrow = "^57.2.0"`, `parquet = "^57.2.0"` already present |
| `las` 0.8.x | `las = "0.9.9"` already present |
| *(not mentioned)* | `polars = "^0.52.0"` is the **actual** Parquet/CSV backend |

`polars` is the heaviest dependency in the graph, yet it is completely absent from the RFC. The existing `src/io/table.rs` uses `polars::prelude::*` for all Parquet and CSV I/O — not `arrow-rs` or `arrow2`. The RFC's claim of "Use `arrow2` (or `arrow-rs`) for zero-copy serialisation" is therefore incorrect today. The RFC must:

1. Acknowledge `polars` as the current backend and justify keeping or replacing it.  
2. Resolve the `arrow2` vs `arrow-rs` ambiguity — `arrow2` is largely unmaintained; `arrow-rs` (Apache) is the recommended choice, and it is already in `Cargo.toml`.  
3. Update all version pins to match the actual `Cargo.toml`.

---

### 4. Existing RGB round-trip bug must be addressed

In `src/io/table.rs` (lines 187–193), RGB channels are **written to the DataFrame as `Vec<u32>`**, but `get_u8_col` (lines 219–242) only recognises `u8`, `u16`, `i32`, and `i64` columns — it will **not** match `u32`. Any Parquet or CSV round-trip that includes RGB data will silently drop the colour channels on read-back.

This is a pre-existing bug, but §3.3 (Full Parquet Round-Trip) explicitly targets attribute preservation, so this RFC owns fixing it. Suggested fix: write RGB as `UInt8` columns directly (convert `u8` → `u8` via `Vec<u8>` Series), matching the schema table proposed in §3.3 which already lists `r`, `g`, `b` as `UInt8`.

---

### 5. Path traversal and security in `IoPlugin` trait

The `IoPlugin` trait (§3.4) declares:

```rust
fn read(&self, path: &str) -> Result<HighPerformancePointCloud>;
fn write(&self, pc: &HighPerformancePointCloud, path: &str) -> Result<()>;
```

Using `&str` instead of `&std::path::Path` loses all platform-aware path validation. More critically, **the plugin registry exposes an arbitrary-write primitive**: a malicious or buggy third-party plugin could write to any path the process has permission to reach (including `../../../etc/cron.d/...` under certain deployments). Recommended mitigations:

- Change signatures to `&Path`.  
- Specify that `detect_format` (§3.1) must canonicalize and validate the path (no `..` components, no symlinks to outside the working tree if a root is configurable) before dispatching to any plugin.  
- Similarly, the existing `path: &str` in `from_las_laz`, `to_las`, `load_from_file`, etc. (`src/io/`) should be audited for path traversal.

---

### 6. No atomic write strategy for large files

If a write is interrupted mid-stream (OOM, SIGKILL, disk full), the output file will be left in a partial and unreadable state — silently overwriting a previously valid file at the same path. For a library targeting 50 GB+ files this is a significant operational risk.

Recommendation: specify that `to_file` / `save_to_file` must write to a sibling `.tmp` file and `rename(2)` on completion. This is a one-syscall atomic operation on POSIX systems and is achievable on Windows via `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`. Add a test case for "interrupted write leaves original file intact."

---

### 7. `IoPlugin` registry concurrency not specified

`register_io_plugin(plugin: Box<dyn IoPlugin>)` is called at runtime, potentially from multiple threads (e.g., two Python extensions registering in parallel `import` statements). The RFC does not specify:

- Whether the registry is a global `OnceLock<RwLock<Vec<...>>>` or similar.  
- Whether registering the same extension twice is an error, a warning, or silently last-wins.  
- Whether plugin lookup during `detect_format` requires a read-lock that can deadlock with a concurrent write.

This must be specified before implementation; a `OnceLock<RwLock<Vec<Box<dyn IoPlugin>>>>` pattern is recommended.

---

### 8. Chunk reader Python API is underspecified

The example in §3.2 uses:

```python
with PointCloud.open("huge_scan.las", chunk_size=1_000_000) as reader:
    for chunk in reader:
        result.to_file(f"out_{i}.parquet")   # ← `i` is never defined
```

The RFC does not describe the Python class `PointCloudChunkReader` at all: its `__enter__`/`__exit__` contract, whether it implements `__iter__`+`__next__` or is a generator, how `StopIteration` is raised, and how errors mid-stream are surfaced. Also, `i` is undefined in the example loop — the code would raise `NameError`. Please fix the example and add a full Python API sketch (similar to how §3.1 shows the full Rust signature).

---

### 9. `chunk_size=0` and last-chunk edge cases not handled

The design does not state what should happen when:

- `chunk_size=0` is passed — this should raise `ValueError` eagerly, not produce an infinite loop.  
- The final chunk has fewer than `chunk_size` points — this is the common case and should be explicitly documented as guaranteed/expected behaviour.  
- The file has 0 points — the iterator should yield zero chunks (not one empty `PointCloud`).

These should be added to both the design prose and the testing plan.

---

### 10. GIL strategy for long-running I/O not addressed

Reading a 50 GB LAS file on a single thread while holding the Python GIL will freeze the entire Python interpreter for the duration. The RFC should specify whether the Rust I/O methods release the GIL via `py.allow_threads(|| ...)` and, if so, how that interacts with the PyO3 `&Python` lifetime requirements in the chunk iterator.

This is particularly important for the chunk reader, where a Python `for` loop will re-acquire/release the GIL on every iteration boundary.

---

### 11. Testing plan gaps

The current plan is a good start but is missing the following cases:

| Missing test | Why it matters |
|---|---|
| Zero-point cloud round-trip (all formats) | Empty files / empty subsets are common edge cases |
| `chunk_size` larger than total points | Must yield exactly one partial chunk, not crash |
| `chunk_size=0` raises `ValueError` | Guard against silent infinite loops |
| File path with Unicode / non-ASCII characters | Common on non-English systems |
| Write to read-only directory raises `PermissionError` | Python users expect OS errors as Python exceptions |
| NaN / Inf values in XYZ coordinates | LAS format does not support NaN; should error, not silently corrupt |
| Interrupted write leaves original file intact | Validates atomic-write requirement from §6 above |
| Magic-byte detection for extension-less files | The design in §3.1 says "reads first 8 bytes if ambiguous" — test it |
| Plugin: same extension registered twice | Registry collision policy |
| Concurrent reads of the same file from multiple threads | Data race / shared file descriptor |
| Rust `cargo test` coverage for streaming code | The plan only mentions pytest; Rust unit tests should also be specified |
| Performance regression benchmark | A >2 GB/s Parquet claim (§Performance Implications) needs a CI benchmark gate |

---

### 12. Missing unresolved questions

The following questions should be added to §Unresolved Questions:

4. **Canonical API name**: should the Python entry points be `from_file`/`to_file` (RFC proposal) or `load_from_file`/`save_to_file` (current code)? Pick one before implementation.  
5. **GIL release policy**: which I/O methods release the GIL, and via what mechanism?  
6. **Atomic write**: is the `.tmp`+rename pattern required for all formats or only for large-file paths?  
7. **`total_points()` for CSV**: CSV has no header with point count; `Option<u64>` will always be `None` for `CsvChunkReader`. Should the Python layer expose a progress indicator that works without a total?  
8. **Plugin collision policy**: last-wins, first-wins, or `Err`?  
9. **`delete_file` scope**: the static `PointCloud.delete_file(path)` method is already exposed on the Python class. Should it remain, or should I/O cleanup be left to the caller?

---

### Minor / Line-Level Notes

- **§3.1 Rust snippet**: `FileFormat::Unknown(ext)` captures the extension as a `String` — but if `detect_format` falls through to magic bytes and still cannot identify the format, it has no extension to report. Ensure the `Unknown` variant can also carry a descriptive message (e.g., `Unknown { ext: Option<String>, magic: Option<[u8;8]> }`).
- **§3.3 schema table**: `x`, `y`, `z` are listed as `Float32` but LAS stores coordinates as `f64` (double) after scale/offset application. Downgrading to `f32` during Parquet write is lossy for high-precision surveys (sub-centimetre accuracy at city scale). Consider making the XYZ type `Float64` or at minimum documenting the precision loss.
- **§3.5**: The `csv` crate is listed as "already present" but the actual CSV implementation in `table.rs` uses `polars` CSV reader, not the `csv` crate directly. Verify whether the `csv` crate dependency is still needed, or remove it to reduce compile times.
