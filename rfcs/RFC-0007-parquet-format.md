# RFC-0007: Parquet Format I/O

| Field   | Value                              |
|---------|------------------------------------|
| Status  | Draft                              |
| Authors | liuzhen19 \<liuzhen19@xiaomi.com\> |
| Created | 2026-05-03                         |
| Updated | 2026-05-03                         |

## Summary

Complete and fully expose Parquet format read/write support for `PointCloud`. A partial scaffold already exists in `src/io/parquet.rs` but is not wired into the PyO3 bindings. This RFC defines the Parquet schema convention, the Python API, and the implementation plan to finish the feature.

## Motivation

Parquet is the de facto standard columnar storage format for large tabular data:

- **Efficient compression**: Snappy/ZSTD column-level compression achieves 3–10× size reduction vs. LAS binary.
- **Ecosystem integration**: directly readable by pandas, polars, DuckDB, Apache Spark, and cloud data warehouses.
- **Schema flexibility**: arbitrary named columns — supports all custom attributes including normals ([RFC-0005](./RFC-0005-normal-estimation.md)) without schema changes.
- **Partitioned datasets**: natural fit for tiled LiDAR archives (one Parquet file per tile).

The current LAS/LAZ format is excellent for raw sensor data but lacks the analytics ecosystem compatibility that Parquet provides.

## Design Details

### Python API

```python
# Write to Parquet
pc.to_parquet("output.parquet", compression="snappy")  # default: snappy
pc.to_parquet("output.parquet", compression="zstd", compression_level=3)
pc.to_parquet("output.parquet", compression="none")

# Read from Parquet
pc = PointCloud.from_parquet("input.parquet")

# Optionally select specific columns (column pruning for large files)
pc = PointCloud.from_parquet(
    "input.parquet",
    columns=["x", "y", "z", "intensity"],  # load only these attributes
)
```

### Parquet Schema Convention

PCL Rustic uses the following column naming convention:

| Column name   | dtype   | Required | Description                          |
|---------------|---------|----------|--------------------------------------|
| `x`           | float32 | ✅ Yes   | X coordinate                         |
| `y`           | float32 | ✅ Yes   | Y coordinate                         |
| `z`           | float32 | ✅ Yes   | Z coordinate                         |
| `intensity`   | float32 | ❌ No    | Intensity / reflectivity             |
| `normal_x`    | float32 | ❌ No    | Normal X component                   |
| `normal_y`    | float32 | ❌ No    | Normal Y component                   |
| `normal_z`    | float32 | ❌ No    | Normal Z component                   |
| `curvature`   | float32 | ❌ No    | Local surface curvature              |
| `r`, `g`, `b` | float32 | ❌ No    | RGB color (0.0–1.0 normalized)       |
| `<name>`      | float32 | ❌ No    | Any custom named attribute           |

**Schema rules**:
1. `x`, `y`, `z` must be present and float32.
2. All other columns are optional float32 columns treated as named attributes.
3. Non-float32 columns are skipped on read with a warning.
4. On write, normals are serialized as `normal_x`, `normal_y`, `normal_z` columns.

### Metadata (Parquet File-Level)

A Parquet key-value metadata entry `pcl_rustic_schema_version = "1"` is written to allow future schema evolution.

### Compression Options

| Option     | Rust feature flag          | Notes                                |
|------------|----------------------------|--------------------------------------|
| `none`     | (always available)         | No compression; fastest read/write   |
| `snappy`   | `parquet/snap` (default)   | Fast, moderate compression ratio     |
| `zstd`     | `parquet/zstd`             | Slow write, excellent ratio; tunable |
| `lz4`      | `parquet/lz4`              | Fastest decompression                |
| `gzip`     | `parquet/gzip`             | High compatibility, slower           |

Default: `snappy` (balances speed and size).

### Row Group Size

Default row group size: 1 M rows. Configurable via `row_group_size` parameter:

```python
pc.to_parquet("output.parquet", row_group_size=500_000)
```

Larger row groups improve compression; smaller improve parallelism for partial reads.

### Streaming / Partitioned Write (Future)

For very large clouds (>100 M pts), a streaming writer API will be provided:

```python
with PointCloud.parquet_writer("output.parquet") as writer:
    for chunk in large_pc.chunks(size=10_000_000):
        writer.write(chunk)
```

This is deferred to a follow-up RFC (RFC-0013).

### Implementation Notes

The existing `src/io/parquet.rs` scaffold uses `parquet` + `arrow` crates (already in `Cargo.toml`). The following changes are needed:

1. **Column mapping**: serialize `HighPerformancePointCloud` fields to Arrow `RecordBatch`.
2. **Schema creation**: build `arrow::datatypes::Schema` from the convention above.
3. **Normals interop**: check for `Option<Vec<[f32; 3]>>` normals field ([RFC-0005](./RFC-0005-normal-estimation.md)) and emit `normal_x/y/z` columns.
4. **Read path**: parse schema, map columns back to struct fields and attribute HashMap.
5. **PyO3 exposure**: add `from_parquet()` classmethod and `to_parquet()` method.

### Module Layout (no structural change)

```
src/io/
├── parquet.rs   # existing file — extend with full read/write logic
├── las_laz.rs
└── csv.rs
```

## Alternatives Considered

### Alternative 1: Arrow IPC (Feather v2) Instead of Parquet

Arrow IPC is faster for in-memory exchange but:
- Less ecosystem support for analytics (DuckDB, Spark prefer Parquet).
- Larger file size than compressed Parquet.

Recommend Arrow IPC as a future lightweight inter-process format (RFC-0014), not as a replacement for Parquet.

### Alternative 2: HDF5

HDF5 is widely used in scientific computing. Rejected because:
- Complex C library dependency (no pure Rust HDF5 at production quality).
- Parquet better suited for cloud-native analytics.

### Alternative 3: Polars-based I/O (Already in Cargo.toml)

`polars` is already a dependency and supports Parquet natively. However, using `polars` for I/O introduces a runtime overhead (polars `DataFrame` intermediate allocation). Direct use of the `parquet` + `arrow` crates is leaner and avoids the double allocation.

## Open Questions

1. **Double/float64 support**: Should Parquet files with float64 columns be automatically cast to float32 on read, or should the library raise an error?
2. **Null handling**: Parquet supports nullable columns. How should null values in `x`, `y`, `z` be handled? (Proposed: raise error; nulls in optional attributes: fill with 0.0 and warn.)
3. **GeoParquet**: The [GeoParquet spec](https://geoparquet.org/) adds geospatial metadata. Should PCL Rustic write GeoParquet-compatible files? (Deferred to RFC-0015.)
4. **Large file streaming**: Is the row-group-based approach sufficient for 500 M+ point clouds, or does a streaming writer need to be in scope for this RFC?

## Implementation Plan

- [ ] Extend `src/io/parquet.rs`: implement `write_parquet(path, compression, row_group_size)`
- [ ] Extend `src/io/parquet.rs`: implement `read_parquet(path, columns)` with column pruning
- [ ] Add normal column serialization (depends on RFC-0005 field layout)
- [ ] Expose `PointCloud.to_parquet()` and `PointCloud.from_parquet()` via PyO3 in `src/lib.rs`
- [ ] Add pytest tests: `tests/test_parquet.py`
  - `test_parquet_roundtrip_xyz()` — write and read back XYZ only
  - `test_parquet_roundtrip_all_attributes()` — write and read back XYZ + intensity + custom attrs
  - `test_parquet_column_pruning()` — `columns=["x","y","z"]` loads only XYZ
  - `test_parquet_compression_snappy()` — snappy compressed file is smaller than uncompressed
  - `test_parquet_compression_zstd()` — zstd compressed file is smaller than snappy
  - `test_parquet_missing_xyz_raises()` — file without `x`, `y`, `z` raises ValueError
  - `test_parquet_schema_metadata()` — written file contains `pcl_rustic_schema_version` metadata
  - `test_parquet_roundtrip_normals()` — normals survive write→read (after RFC-0005)
- [ ] Update `rfcs/README.md` index status
- [ ] Update `README.md` I/O format table (Parquet: ✅ read / ✅ write)

## Related RFCs

- **Background**: [RFC-0001](./RFC-0001-initial-architecture.md) — I/O architecture and file format scaffold
- **Extended by** ← [RFC-0005](./RFC-0005-normal-estimation.md): normals column in Parquet schema
- **Future** → RFC-0013 (planned): Streaming/partitioned Parquet write
- **Future** → RFC-0014 (planned): Arrow IPC (Feather v2) format
- **Future** → RFC-0015 (planned): GeoParquet compatibility

## References

- [Apache Parquet specification](https://parquet.apache.org/docs/file-format/)
- [GeoParquet specification](https://geoparquet.org/)
- [parquet Rust crate](https://docs.rs/parquet/)
- [arrow Rust crate](https://docs.rs/arrow/)
- [polars Rust crate](https://docs.rs/polars/)
- [RFC-0001](./RFC-0001-initial-architecture.md)
- [RFC-0005](./RFC-0005-normal-estimation.md)
