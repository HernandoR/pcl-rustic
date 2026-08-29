"""Device roster and diagnostics.

Regression cover for a silent-fallback bug: on a host whose GPU fails to
initialize, `available_devices()` simply omits it, which is
indistinguishable from a host with no GPU. `device_report()` exists to make
that case diagnosable, so these tests pin its contract rather than asserting
any particular device is present -- the suite has to pass on CPU-only CI too.
"""

from __future__ import annotations

import numpy as np
import pcl_rustic as pcl
import pytest

# Every device name the report is required to mention, regardless of whether
# this build or this host can actually use it.
ALL_DEVICE_NAMES = {"cpu", "metal", "vulkan", "cuda", "torch"}


def test_cpu_is_always_available() -> None:
    assert "cpu" in pcl.available_devices()


def test_default_device_is_available() -> None:
    assert pcl.default_device() in pcl.available_devices()


def test_report_covers_every_device_name() -> None:
    assert set(pcl.device_report()) == ALL_DEVICE_NAMES


def test_report_agrees_with_available_devices() -> None:
    """The two views must not disagree: anything reported `available` has to
    show up in the roster, and vice versa."""
    report = pcl.device_report()
    available = set(pcl.available_devices())
    assert {n for n, s in report.items() if s == "available"} == available


def test_report_explains_unavailable_devices() -> None:
    """An unusable device must carry a reason, not just be absent."""
    for name, status in pcl.device_report().items():
        if status == "available":
            continue
        assert status.startswith(("unavailable: ", "not compiled into this build")), (
            f"{name}: unexplained status {status!r}"
        )
        # A bare prefix with no detail would defeat the point.
        if status.startswith("unavailable: "):
            assert status.removeprefix("unavailable: ").strip()


def test_cpu_is_never_reported_unavailable() -> None:
    assert pcl.device_report()["cpu"] == "available"


def test_unknown_device_name_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        pcl.PointCloud.from_xyz(np.zeros((4, 3))).to_device("nope")


def test_unusable_device_error_carries_the_reason() -> None:
    """`to_device` on a compiled-in but unusable device must explain why,
    rather than failing with a bare "not usable on this machine"."""
    report = pcl.device_report()
    unusable = [name for name, s in report.items() if s.startswith("unavailable: ")]
    if not unusable:
        pytest.skip("every compiled-in device is usable on this host")
    cloud = pcl.PointCloud.from_xyz(np.zeros((4, 3)))
    with pytest.raises(OSError) as excinfo:
        cloud.to_device(unusable[0])
    # The underlying backend message, not just our wrapper text.
    assert len(str(excinfo.value)) > len("device 'x' is compiled in but not usable")


def test_round_trip_on_every_available_device() -> None:
    xyz = np.arange(12, dtype=np.float64).reshape(4, 3)
    for name in pcl.available_devices():
        cloud = pcl.PointCloud.from_xyz(xyz).to_device(name)
        assert cloud.device == name
        # f32-relative storage on GPU devices costs precision, so this is a
        # tolerance check, not an exact one (ADR-0002).
        np.testing.assert_allclose(cloud.xyz, xyz, rtol=1e-6, atol=1e-6)
