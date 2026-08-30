"""Comparison benchmarks: pcl-rustic vs Open3D, PCL, and laspy.

Run with `just bench-compare` (Open3D + laspy) or `just bench-pcl` (adds the
PCL baseline). Every baseline is optional; missing ones skip.

Baseline sourcing
-----------------
Open3D and laspy are ordinary Python packages in the `test` dependency group.
PCL is **not**, because it has no working Python binding on modern Python:
`python-pcl`'s newest release (0.3.0a1, 2018) ships only cp27/cp35 wheels and
its `setup.py` hard-codes PCL 1.7/1.8/1.9 branches -- against PCL 1.14 it
falls through to `else: pass` and dies with
`NameError: name 'vtk_include_dir' is not defined`. `pclpy` pins
`requires_python == 3.6.*` with a Windows-only wheel. The PCL baseline is
therefore a small C++ harness (`bench/pcl_cpp/`) invoked from this module, so
the comparison is still driven and reported from Python.

Fairness notes -- read these before quoting any number
------------------------------------------------------
* All engines consume byte-identical input: the C++ harness reads the same
  f64 xyz dump written from the very array handed to Open3D and pcl-rustic.
* Cloud construction is outside every timed region, on all sides.
* **Voxel semantics differ.** Open3D's `voxel_down_sample` and PCL's
  `VoxelGrid` emit a synthetic averaged point per voxel; pcl-rustic's
  `NEAREST_TO_CENTROID` picks the nearest *existing* point, which costs an
  extra distance pass. Output counts are printed so the difference is
  visible. This compares a user task, not an identical kernel.
* **Transform result materialization differs.** Open3D transforms in place
  and `np.asarray(pcd.points)` / `.numpy()` are zero-copy views; PCL writes
  into a second f32 cloud. pcl-rustic's transform therefore also runs
  `inplace=True` in every comparison here (matching Open3D's semantics),
  but its `.xyz` still allocates and copies a fresh array where Open3D
  hands out a view -- pcl-rustic remains charged for work the others are
  not wherever a timed region includes materialization.
* **Element types differ by design.** PCL `PointXYZ` is f32, Open3D is f64,
  pcl-rustic is f32 relative to an f64 offset.
* **The device is printed, and it changes the meaning of every row.** These
  baselines are all CPU. When pcl-rustic runs on a GPU the transform row is
  dominated by PCIe readback and the f32 -> f64 widening, not by the
  transform: measured on an A10G at n=10M, submit is ~0.3 ms and
  materialization ~260 ms, versus ~31 ms and ~161 ms on `cpu`. So the GPU
  can be *slower* here while doing the arithmetic ~100x faster. Compare
  like-for-like by pinning the device before drawing conclusions.
* **The four-way end-to-end tests have their own semantics.** They compare
  pcl-rustic and Open3D on CPU *and* GPU simultaneously, and every engine's
  timed region is op + materializing the result positions to host numpy --
  device readback deliberately included, unlike the compute-only GPU rows
  above. Open3D's GPU column uses the tensor API (`open3d.t`) on CUDA, so it
  exercises a different Open3D code path than the legacy-API CPU rows; its
  `.numpy()` on a CPU tensor is a zero-copy view while pcl-rustic's `.xyz`
  always allocates and copies. Positions are float32 in the four base rows
  (pcl-rustic's default dtype). The fifth row, `pcl-rustic[cpu f64 view]`,
  is the true zero-copy-for-zero-copy comparison against `open3d[cpu]`:
  float64 CPU cloud, in-place op, readout via `view()` -- the one
  configuration where neither engine copies its result.
"""

from __future__ import annotations

import statistics
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import resolve_bench_device
from pcl_rustic import (
    DownsampleStrategy,
    PointCloud,
    available_devices,
    get_default_dtype,
    set_default_dtype,
)

o3d = pytest.importorskip("open3d", reason="open3d not installed")
laspy = pytest.importorskip("laspy", reason="laspy not installed")
o3c = o3d.core

pytestmark = [pytest.mark.bench, pytest.mark.slow]

PCL_BIN = (
    Path(__file__).resolve().parents[1] / "bench" / "pcl_cpp" / "build" / "pcl_bench"
)

POINT_COUNTS = [1_000_000, 10_000_000]
VOXEL_SIZE = 0.15
# PCL's VoxelGrid overflows its int32 voxel index at VOXEL_SIZE over this
# extent, so the three-way run uses a leaf it accepts. The refusal is
# exercised on its own below.
PCL_SAFE_LEAF = 1.0
PCL_POINTS = 2_000_000
IO_POINTS = 2_000_000


def _gaussian_xyz(n_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=100.0, size=(n_points, 3))


BENCH_DEVICE = resolve_bench_device()
GPU_MODE = BENCH_DEVICE != "cpu"


@pytest.fixture(scope="module", autouse=True)
def _device_banner() -> None:
    """Name the device and dtype up front: every pcl-rustic number below
    depends on both, while every baseline is CPU. Without this a run that
    silently fell back to `cpu` reads identically to a GPU run."""
    mode = " (op timing: compute only, no readback)" if GPU_MODE else ""
    print(
        f"\n[pcl-rustic device: {BENCH_DEVICE}  dtype: {get_default_dtype()}"
        f"  |  baselines: cpu{mode}]"
    )


def _pc(xyz: np.ndarray) -> PointCloud:
    return PointCloud.from_xyz(xyz).to_device(BENCH_DEVICE)


def _timed_transform(pc: PointCloud, matrix: np.ndarray) -> float:
    """Median wall time of one transform, honoring the device's semantics.

    On the GPU the timed region is submit + `synchronize()`: kernel
    execution without the PCIe readback and numpy materialization, which are
    boundary costs paid once per host readout, not per op. On the CPU it is
    op + `.xyz`, matching what the CPU baselines are charged for (their
    results are host-resident and materialized by construction).
    """
    # inplace=True on both branches: Open3D's transform is in place, so the
    # copying path would charge pcl-rustic a dims clone and a coordinate
    # allocation per call that the baseline never pays. Calls compose, as
    # Open3D's do; per-call cost is unchanged.
    if GPU_MODE:

        def op() -> PointCloud:
            out = pc.transform(matrix, inplace=True)
            out.synchronize()
            return out
    else:

        def op() -> np.ndarray:  # type: ignore[misc]
            return pc.transform(matrix, inplace=True).xyz

    elapsed, _ = _timed(op)
    return elapsed


#: One warmup run, then the median of this many timed runs. Without the
#: warmup, pcl-rustic's first op in a process pays a one-time kernel-init
#: cost (~55 ms), which made results depend on test ordering.
REPEAT = 3


def _timed(fn, repeat: int = REPEAT):
    fn()  # warmup
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result


def _verdict(ours: float, theirs: float) -> str:
    ratio = theirs / ours if ours > 0 else float("inf")
    return f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"


def _row(
    op: str, n: int, ours: float, theirs: float, baseline: str, extra: str = ""
) -> None:
    print(
        f"{op:<26} n={n:>11,}  pcl-rustic={ours:7.3f}s  {baseline}={theirs:7.3f}s"
        f"  -> pcl-rustic is {_verdict(ours, theirs)}{extra}"
    )


def _o3d_cloud(xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def _transform_matrix() -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    return matrix


# -- PCL C++ harness bridge ---------------------------------------------


def _require_pcl() -> None:
    if not PCL_BIN.exists():
        pytest.skip(f"PCL harness not built ({PCL_BIN}); run `just bench-pcl`")


def _dump_xyz(xyz: np.ndarray, path: Path) -> None:
    with path.open("wb") as fh:
        fh.write(struct.pack("<q", xyz.shape[0]))
        fh.write(np.ascontiguousarray(xyz, dtype="<f8").tobytes())


def _run_pcl(path: Path, leaf: float) -> dict[str, float]:
    proc = subprocess.run(
        [str(PCL_BIN), str(path), str(leaf), str(REPEAT)],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = dict(tok.split("=", 1) for tok in proc.stdout.split())
    return {k: float(v) for k, v in fields.items()}


# -- Open3D comparisons --------------------------------------------------


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_vs_open3d_voxel_downsample(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    pc = _pc(xyz)
    pcd = _o3d_cloud(xyz)

    ours, down = _timed(
        lambda: pc.voxel_downsample(
            VOXEL_SIZE, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
        )
    )
    theirs, down_o3d = _timed(lambda: pcd.voxel_down_sample(VOXEL_SIZE))

    _row(
        "voxel_downsample",
        n_points,
        ours,
        theirs,
        "open3d",
        f"  (out: {len(down):,} vs {len(down_o3d.points):,})",
    )
    assert 0 < len(down) <= n_points


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_vs_open3d_transform(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points)
    matrix = _transform_matrix()
    pc = _pc(xyz)
    pcd = _o3d_cloud(xyz)

    # Open3D's transform mutates in place, so repeated calls compose. The
    # per-call work is identical either way, so the timing stays valid, but
    # the numerical check has to run on untouched clouds (below).
    ours = _timed_transform(pc, matrix)
    theirs, _ = _timed(lambda: np.asarray(pcd.transform(matrix).points))

    _row("transform (4x4)", n_points, ours, theirs, "open3d")

    # Both must agree on the math, or the comparison is meaningless: one
    # application each, on fresh clouds.
    fresh_ours = _pc(xyz).transform(matrix).xyz
    fresh_theirs = np.asarray(_o3d_cloud(xyz).transform(matrix).points)
    np.testing.assert_allclose(fresh_ours, fresh_theirs, atol=1e-3)


# -- PCL comparisons -----------------------------------------------------


def test_vs_pcl_cpp(tmp_path: Path) -> None:
    _require_pcl()
    xyz = _gaussian_xyz(PCL_POINTS)
    data = tmp_path / "cloud.bin"
    _dump_xyz(xyz, data)

    pcl = _run_pcl(data, PCL_SAFE_LEAF)
    assert not pcl["voxel_refused"], "PCL refused the leaf size; comparison invalid"

    pc = _pc(xyz)
    ours_voxel, down = _timed(
        lambda: pc.voxel_downsample(
            PCL_SAFE_LEAF, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
        )
    )
    ours_transform = _timed_transform(pc, _transform_matrix())
    moved = pc.transform(_transform_matrix()).xyz

    _row(
        f"voxel_downsample (leaf {PCL_SAFE_LEAF})",
        PCL_POINTS,
        ours_voxel,
        pcl["voxel_secs"],
        "pcl-c++",
        f"  (out: {len(down):,} vs {int(pcl['voxel_out']):,})",
    )
    _row(
        "transform (4x4)", PCL_POINTS, ours_transform, pcl["transform_secs"], "pcl-c++"
    )
    # Near-identical voxel binning, independent of speed. Not exactly equal:
    # PCL's PointXYZ is f32 while our CPU coordinates are exact f64, so points
    # sitting on a voxel boundary can round to different cells. The residual
    # is a couple of points in ~2M.
    theirs_out = int(pcl["voxel_out"])
    assert abs(len(down) - theirs_out) <= max(8, theirs_out // 100_000), (
        f"voxel counts diverge beyond f32/f64 boundary rounding: {len(down)} vs {theirs_out}"
    )
    assert moved.shape == (PCL_POINTS, 3)


def test_pcl_voxelgrid_refuses_fine_leaf(tmp_path: Path) -> None:
    """PCL silently no-ops where pcl-rustic downsamples; keep that documented."""
    _require_pcl()
    xyz = _gaussian_xyz(PCL_POINTS)
    data = tmp_path / "cloud.bin"
    _dump_xyz(xyz, data)

    pcl = _run_pcl(data, VOXEL_SIZE)
    pc = _pc(xyz)
    down = pc.voxel_downsample(
        VOXEL_SIZE, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
    )

    print(
        f"leaf {VOXEL_SIZE} over a +/-500 m extent: PCL returned "
        f"{int(pcl['voxel_out']):,} points (refused={bool(pcl['voxel_refused'])}), "
        f"pcl-rustic returned {len(down):,}"
    )
    assert pcl["voxel_refused"] == 1.0
    assert int(pcl["voxel_out"]) == PCL_POINTS
    assert len(down) < PCL_POINTS


# -- laspy comparisons ---------------------------------------------------


def test_vs_laspy_read(tmp_path: Path) -> None:
    xyz = _gaussian_xyz(IO_POINTS, seed=3)
    rng = np.random.default_rng(4)

    source = PointCloud.from_xyz(xyz)
    source["intensity"] = rng.integers(0, 65535, size=IO_POINTS).astype(np.uint16)
    source["gps_time"] = rng.normal(size=IO_POINTS) * 1.0e3
    path = tmp_path / "bench.las"
    source.write(str(path))

    def theirs_op() -> np.ndarray:
        las = laspy.read(str(path))
        return np.column_stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
        )

    loaded = PointCloud.read(str(path))
    ours, ours_xyz = _timed(lambda: PointCloud.read(str(path)).xyz)
    theirs, theirs_xyz = _timed(theirs_op)

    # The zero-copy pipeline: read at float64 (CPU-resident) and hand out a
    # view instead of a copying .xyz. The view keeps its cloud's buffer
    # alive, so the temporary cloud may be dropped.
    previous = get_default_dtype()
    set_default_dtype("float64")
    try:
        ours_view, ours_view_xyz = _timed(lambda: PointCloud.read(str(path)).view())
    finally:
        set_default_dtype(previous)

    _row("LAS read + xyz", IO_POINTS, ours, theirs, "laspy ")
    _row("LAS read + view (f64)", IO_POINTS, ours_view, theirs, "laspy ")
    # f64/CPU is the exact path: bit-identical to laspy, via the view too.
    assert float(np.abs(ours_view_xyz - theirs_xyz).max()) == 0.0
    las_scale = 1.0e-3
    max_dev = float(np.abs(np.asarray(ours_xyz, dtype=np.float64) - theirs_xyz).max())
    print(
        f"{'  coordinate fidelity':<26} [{loaded.device}/{loaded.dtype}] "
        f"max deviation vs laspy = {max_dev:.3e} m "
        f"({max_dev / las_scale:.1%} of the {las_scale} LAS scale step)"
    )

    # ADR-0002 promises different things depending on where the cloud landed
    # and its coordinate dtype; `read` uses the ambient defaults. Asserting
    # the strictest guarantee unconditionally only passed while GPU init was
    # silently failing.
    if loaded.device == "cpu" and loaded.dtype == np.float64:
        # CPU-resident f64 clouds hold absolute f64, so a LAS round-trip is
        # bit-identical to laspy's: both compute offset + raw * scale in f64.
        assert max_dev == 0.0, (
            f"expected bit-identical coordinates on CPU/f64, got deviation {max_dev}"
        )
    else:
        # f32-relative representations (GPU, or the float32 default dtype on
        # CPU) promise only that the error stays under LAS quantization,
        # i.e. the round-trip loses nothing the format itself preserved.
        assert max_dev < las_scale / 2, (
            f"deviation {max_dev} on '{loaded.device}'/{loaded.dtype} exceeds "
            f"half the LAS scale step ({las_scale / 2}); f32-relative "
            f"coordinates are supposed to stay under quantization"
        )


def test_vs_laspy_dtype_parity(tmp_path: Path) -> None:
    """The dtype-promise claim, checked against laspy on a real file."""
    xyz = _gaussian_xyz(1000, seed=5)
    rng = np.random.default_rng(6)

    source = PointCloud.from_xyz(xyz)
    source["intensity"] = rng.integers(0, 65535, size=1000).astype(np.uint16)
    source["classification"] = rng.integers(0, 11, size=1000).astype(np.uint8)
    path = tmp_path / "dtype.las"
    source.write(str(path))

    ours = PointCloud.read(str(path))
    theirs = laspy.read(str(path))

    for dim in ("intensity", "classification"):
        mine = ours[dim]
        yours = np.asarray(getattr(theirs, dim))
        assert mine.dtype == yours.dtype, (
            f"{dim}: {mine.dtype} != laspy's {yours.dtype}"
        )
        np.testing.assert_array_equal(mine, yours)
    print(
        "dtype parity with laspy: intensity="
        f"{ours['intensity'].dtype}, classification={ours['classification'].dtype} (exact match)"
    )


# -- four-way end-to-end: pcl-rustic vs Open3D, CPU vs GPU ----------------
#
# Unlike the rows above (which pin pcl-rustic to BENCH_DEVICE and, on GPU,
# time compute only), these run every engine/device combination available on
# this host side by side, and every timed region is END-TO-END: the op plus
# materializing the result positions to host numpy, device readback
# included. See the module docstring's fairness notes.


def _four_way_engines() -> list[tuple[str, str, str]]:
    """(label, engine, device) for every combination this host supports.

    `pcl64` is the zero-copy configuration: float64 CPU cloud whose readout
    is a `view()` rather than a copying `.xyz`, matching Open3D's zero-copy
    `.numpy()` semantics on the result side.
    """
    engines = [
        ("pcl-rustic[cpu]", "pcl", "cpu"),
        ("pcl-rustic[cpu f64 view]", "pcl64", "cpu"),
    ]
    gpus = [d for d in available_devices() if d != "cpu"]
    if gpus:
        engines.append((f"pcl-rustic[{gpus[0]}]", "pcl", gpus[0]))
    engines.append(("open3d[cpu]", "o3d", "CPU:0"))
    if o3c.cuda.is_available():
        engines.append(("open3d[cuda]", "o3d", "CUDA:0"))
    return engines


def _pcl64_cloud(xyz: np.ndarray) -> PointCloud:
    """A float64 (hence CPU-resident) cloud, restoring the ambient default."""
    previous = get_default_dtype()
    set_default_dtype("float64")
    try:
        return PointCloud.from_xyz(xyz)
    finally:
        set_default_dtype(previous)


def _o3d_t_cloud(xyz: np.ndarray, device: str):
    return o3d.t.geometry.PointCloud(
        o3c.Tensor(
            np.ascontiguousarray(xyz, dtype=np.float32), device=o3c.Device(device)
        )
    )


def _e2e_report(op: str, n: int, results: list[tuple[str, float]]) -> None:
    base = dict(results).get("open3d[cpu]")
    print(f"{op} end-to-end   n={n:>11,}   (op + host materialization)")
    for label, secs in results:
        vs = f"  {base / secs:5.2f}x vs open3d[cpu]" if base is not None else ""
        print(f"  {label:<26} {secs:8.3f}s  {n / secs:>15,.0f} pts/s{vs}")


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_four_way_transform_end_to_end(n_points: int) -> None:
    xyz = _gaussian_xyz(n_points).astype(np.float32)
    matrix = _transform_matrix()

    def pcl_op(device: str):
        pc = PointCloud.from_xyz(xyz).to_device(device)

        # inplace, matching Open3D's semantics below; both compose.
        def op() -> np.ndarray:
            return pc.transform(matrix, inplace=True).xyz

        return op

    def pcl64_op():
        pc = _pcl64_cloud(xyz)

        # Zero-copy readout: view() materializes nothing. The full view is
        # a temporary dropped inside the op (only a one-row copy escapes),
        # so the next iteration's in-place transform is not forced through
        # a copy-on-write split by a still-live view.
        def op() -> np.ndarray:
            return pc.transform(matrix, inplace=True).view()[:1].copy()

        return op

    def o3d_op(device: str):
        pcd = _o3d_t_cloud(xyz, device)
        t = o3c.Tensor(matrix, device=o3c.Device(device))

        # In place, so calls compose; per-call cost is unchanged and the
        # math is cross-checked in test_vs_open3d_transform.
        def op() -> np.ndarray:
            return pcd.transform(t).point.positions.cpu().numpy()

        return op

    ops = {"pcl": pcl_op, "o3d": o3d_op}
    results = []
    for label, engine, device in _four_way_engines():
        secs, out = _timed(pcl64_op() if engine == "pcl64" else ops[engine](device))
        assert out.ndim == 2 and out.shape[1] == 3
        results.append((label, secs))

    _e2e_report("transform (4x4)", n_points, results)


@pytest.mark.parametrize("n_points", POINT_COUNTS)
def test_four_way_voxel_end_to_end(n_points: int) -> None:
    # Voxel semantics differ (Open3D averages per voxel, pcl-rustic keeps
    # the nearest existing point) -- this compares the user task, not an
    # identical kernel; output counts are asserted close, not equal.
    xyz = _gaussian_xyz(n_points).astype(np.float32)

    def pcl_op(device: str):
        pc = PointCloud.from_xyz(xyz).to_device(device)

        def op() -> np.ndarray:
            return pc.voxel_downsample(
                VOXEL_SIZE, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
            ).xyz

        return op

    def pcl64_op():
        pc = _pcl64_cloud(xyz)

        # The result is a fresh f64/CPU cloud each call, so its view() is
        # zero-copy with no copy-on-write interaction across iterations.
        def op() -> np.ndarray:
            return pc.voxel_downsample(
                VOXEL_SIZE, strategy=DownsampleStrategy.NEAREST_TO_CENTROID
            ).view()

        return op

    def o3d_op(device: str):
        pcd = _o3d_t_cloud(xyz, device)

        def op() -> np.ndarray:
            return pcd.voxel_down_sample(VOXEL_SIZE).point.positions.cpu().numpy()

        return op

    ops = {"pcl": pcl_op, "o3d": o3d_op}
    results = []
    counts = {}
    for label, engine, device in _four_way_engines():
        secs, out = _timed(pcl64_op() if engine == "pcl64" else ops[engine](device))
        counts[label] = out.shape[0]
        results.append((label, secs))

    _e2e_report("voxel_downsample", n_points, results)
    print(f"  output counts: { {k: f'{v:,}' for k, v in counts.items()} }")
    # Same voxel grid resolution everywhere: counts agree within f32
    # boundary-rounding noise.
    low, high = min(counts.values()), max(counts.values())
    assert high - low <= max(8, high // 100_000), f"voxel counts diverge: {counts}"
