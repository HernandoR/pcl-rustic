//! Runtime device selection for the unified `burn::Dispatch` backend
//! (ADR-0001).
//!
//! Device roster (see Cargo.toml for the wiring):
//!
//! | device name | burn feature | `DispatchDevice` variant | availability |
//! |---|---|---|---|
//! | `cpu`    | `burn/flex`   | `Flex` (pure-Rust CPU backend) | always |
//! | `metal`  | `burn/metal`  | `Metal` (wgpu/Metal, MSL) | macOS targets |
//! | `vulkan` | `burn/vulkan` | `Vulkan` (wgpu/Vulkan, SPIR-V) | non-macOS targets |
//! | `cuda`   | `burn/cuda`   | `Cuda` | `cuda` feature |
//! | `torch`  | `burn/tch`    | `LibTorch` | `torch` feature |
//!
//! `cpu` and exactly one wgpu flavour are always compiled in, so every wheel
//! carries a GPU backend for its platform and the choice of device stays a
//! runtime one. `metal` and `vulkan` are gated on `target_os` rather than on
//! cargo features because burn-dispatch's wgpu flavours are mutually
//! exclusive *and* fail silently when combined -- see the Cargo.toml comment.
//! The `target_os` gates here therefore mirror the manifest exactly: on macOS
//! `DispatchDevice::Vulkan` does not exist (burn-dispatch cfg-gates the
//! variant on `wgpu_vulkan`), so a mismatch is a compile error, not a
//! silently dead device.
//!
//! `cpu` maps to `burn/flex` (the pure-Rust CPU backend). The CubeCL
//! `burn/cpu` backend was tried first per ADR-0001: it compiles cleanly, but
//! aborts at runtime on tall matmuls. Reproduced deterministically: a
//! `[N,3] x [3,3]` f32 matmul on `DispatchDevice::Cpu` overflows the stack of
//! an internal cubecl worker thread for N >= ~5.6M (N = 5M completes in
//! 2.3s); the crash is not avoidable by growing the caller's stack, since the
//! overflowing thread is spawned internally. Same failure on burn 0.21.0 and
//! 0.22.0-pre.2. `flex` runs the identical [10M,3] workload in 0.22s.
//! Upstream issue: <https://github.com/tracel-ai/burn/issues/5419>.
//!
//! That crash is now **fixed upstream** by
//! <https://github.com/tracel-ai/burn/pull/5423> (merged 2026-08-26): the
//! `gemv` autotune group was handing `PRIORITY_MAX` to 1D vector routines for
//! any CPU matmul, so a tall `[N,3] x [3,3]` was scheduled as N one-element
//! micro-cubes and exhausted the worker stack during partition bisection. The
//! fix gates that on `MatmulKind::{MatVec, VecMat}`.
//!
//! We are nonetheless **staying on `flex`**, for two independent reasons:
//!
//! 1. The fix is not in any published release. The newest crates.io version
//!    is 0.22.0-pre.3 (2026-08-25), which predates the merge by a day, so
//!    adopting it would mean a git pin -- unjustifiable for a backend we do
//!    not otherwise need.
//! 2. Even fixed, it is slower for our shape. The PR reports the repaired
//!    path routing to `CpuGemmStrategy` at ~3.4s for the workload `flex`
//!    already does in 0.22s -- a ~15x regression, because the fix restores
//!    correctness, not competitiveness, on extreme aspect ratios.
//!
//! So this is worth *re-checking* once a release carries the fix (and
//! benchmarking `[10M,3] x [3,3]` before switching), but the crash was only
//! ever the tiebreaker; the throughput gap is the actual reason.
//!
//! `torch` also declares a direct optional dependency on `burn-dispatch`
//! itself (see Cargo.toml). This works around a wiring gap in burn 0.21:
//! every other backend feature on the `burn` facade crate (`cpu`, `cuda`,
//! `vulkan`, `metal`, `webgpu`, `rocm`, `flex`, `ndarray`) forwards to the
//! same-named feature on `burn-dispatch`, but `tch` does not (verified by
//! reading `burn-0.21.0/Cargo.toml`: its `tch` feature only pulls in
//! `burn-tch`, never `burn-dispatch?/tch`). Enabling `burn/tch` alone would
//! therefore compile `burn-tch` without ever compiling `DispatchDevice`'s
//! `LibTorch` variant. Our phantom dependency requests `burn-dispatch`'s own
//! `tch` feature directly; Cargo unifies it with the `burn-dispatch` instance
//! pulled in transitively by `burn`, so the variant appears project-wide. We
//! never reference the `burn_dispatch` crate by name in code -- everything
//! goes through `burn::Dispatch` / `burn::DispatchDevice`. This gap is fixed
//! upstream in burn 0.22, where `burn/tch` reaches `burn-dispatch/tch` via
//! `burn-core/tch` -> `burn-tensor/tch`; drop the phantom dep on upgrade.

use burn::tensor::Tensor;
use burn::Dispatch;
use std::sync::{Arc, LazyLock, Mutex};

pub use burn::DispatchDevice;

use crate::error::{Error, Result};

/// The single tensor backend used throughout the crate (ADR-0001).
pub type B = Dispatch;

/// Attempts to construct the `DispatchDevice` for a given name. Returns
/// `None` when the device is not part of this build: either its cargo
/// feature is off, or it belongs to another target's wgpu flavour.
fn construct(name: &str) -> Option<DispatchDevice> {
    match name {
        "cpu" => Some(DispatchDevice::Flex(Default::default())),
        #[cfg(target_os = "macos")]
        "metal" => Some(DispatchDevice::Metal(Default::default())),
        #[cfg(not(target_os = "macos"))]
        "vulkan" => Some(DispatchDevice::Vulkan(Default::default())),
        #[cfg(feature = "cuda")]
        "cuda" => Some(DispatchDevice::Cuda(Default::default())),
        #[cfg(feature = "torch")]
        "torch" => Some(DispatchDevice::LibTorch(Default::default())),
        _ => None,
    }
}

/// Device names in GPU-first preference order, used by `default_device`.
/// `cpu` is intentionally last: it is the universal fallback. `metal` and
/// `vulkan` are both listed because only one of them can construct on any
/// given target; the other is filtered out by `construct`.
const PREFERENCE_ORDER: &[&str] = &["cuda", "metal", "vulkan", "torch", "cpu"];

/// Every device name any build knows how to construct, regardless of whether
/// this one actually can (`construct` filters that).
const ALL_NAMES: &[&str] = &["cpu", "metal", "vulkan", "cuda", "torch"];

/// Attempts a trivial tensor round-trip on `device`, catching the panics
/// that GPU backends raise when no matching adapter is present at runtime.
/// This is deliberately distinct from compile-time feature availability:
/// a Vulkan build on a headless CI box compiles fine but has no adapter.
///
/// On failure the panic message is returned rather than discarded, so
/// [`device_report`] and [`parse_device_name`] can say *why* a GPU that is
/// physically present did not come up. The common Linux case is a host with
/// an NVIDIA driver (hence an ICD at `/etc/vulkan/icd.d/`) but no
/// `libvulkan.so.1` loader package installed: wgpu then enumerates zero
/// adapters and we silently degrade to `cpu`, which is exactly the failure
/// that is impossible to diagnose from `available_devices()` alone.
///
/// The panic hook is process-global, so a concurrent panic on an unrelated
/// thread can be swallowed (or misattributed) during the probe window. That
/// race predates the message capture and is accepted: probing happens once,
/// early, behind `DEFAULT_DEVICE`'s `LazyLock`.
fn probe_detail(device: &DispatchDevice) -> std::result::Result<(), String> {
    let device = device.clone();
    // GPU backends panic deep inside cubecl worker threads when no adapter
    // exists; every thread's panic goes through the global hook, so replace
    // it for the probe window both to keep headless-host fallback quiet and
    // to capture the reason from whichever thread actually failed --
    // `catch_unwind` below only ever sees the caller thread's payload.
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&captured);
    let saved_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // First panic wins: the deepest one is the informative one, and any
        // later panic is usually just the failure cascading back out.
        let mut slot = sink.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(info.to_string());
        }
    }));
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let tensor = Tensor::<B, 1>::zeros([1], &device);
        let _ = tensor.into_data();
    }))
    .is_ok();
    std::panic::set_hook(saved_hook);
    if ok {
        return Ok(());
    }
    let reason = captured
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_else(|| "backend initialization failed without a message".to_string());
    Err(reason)
}

/// [`probe_detail`] reduced to a yes/no, for the callers that only branch on it.
fn probe(device: &DispatchDevice) -> bool {
    probe_detail(device).is_ok()
}

/// Per-device status for every name in [`ALL_NAMES`], in roster order:
/// `"available"`, `"not compiled into this build"`, or `"unavailable: <panic
/// message>"`. This is the diagnostic counterpart to `available_devices()`,
/// which reports *that* a GPU is missing but never *why* -- turning a silent
/// CPU fallback into something a user can act on.
pub fn device_report() -> Vec<(String, String)> {
    ALL_NAMES
        .iter()
        .map(|name| {
            let status = match construct(name) {
                None => "not compiled into this build".to_string(),
                // `cpu` is never probed, matching `available_devices`.
                Some(_) if *name == "cpu" => "available".to_string(),
                Some(device) => match probe_detail(&device) {
                    Ok(()) => "available".to_string(),
                    Err(reason) => format!("unavailable: {reason}"),
                },
            };
            ((*name).to_string(), status)
        })
        .collect()
}

static DEFAULT_DEVICE: LazyLock<DispatchDevice> = LazyLock::new(|| {
    for name in PREFERENCE_ORDER {
        if let Some(device) = construct(name) {
            if *name == "cpu" || probe(&device) {
                return device;
            }
        }
    }
    // `cpu` is always compiled in (burn's `flex` feature is pinned by both
    // target dependency tables), so this is unreachable unless the manifest
    // is broken.
    construct("cpu").expect("the flex CPU backend was not compiled in")
});

/// The default compute device: the first GPU variant compiled in whose
/// runtime probe succeeds, falling back to `cpu`. Probing failure degrades
/// gracefully -- never a panic at import (ADR-0001). Cached after first use.
pub fn default_device() -> DispatchDevice {
    DEFAULT_DEVICE.clone()
}

/// The always-available pure-Rust CPU device, without going through the
/// probe machinery. Float64-dtype clouds ingest here regardless of
/// [`default_device`]: no accelerator representation can hold f64, so
/// placing a cloud whose dtype promises f64 on a GPU would round it to f32
/// at construction -- exactly what choosing f64 asks to avoid.
pub fn cpu_device() -> DispatchDevice {
    DispatchDevice::Flex(Default::default())
}

/// Every device name usable on this machine right now: compiled-in variants
/// that pass the runtime probe. `cpu` is always listed when compiled in,
/// without probing (ADR-0001): CPU tensor creation cannot meaningfully fail
/// the way an absent GPU adapter can.
pub fn available_devices() -> Vec<String> {
    ALL_NAMES
        .iter()
        .filter_map(|name| {
            let device = construct(name)?;
            (*name == "cpu" || probe(&device)).then(|| (*name).to_string())
        })
        .collect()
}

/// Parses a device name (one of [`ALL_NAMES`]) into a `DispatchDevice`,
/// probing it (except `cpu`) so a caller asking for a specific device gets a
/// clear error instead of a later panic.
pub fn parse_device_name(name: &str) -> Result<DispatchDevice> {
    if !ALL_NAMES.contains(&name) {
        return Err(Error::value(format!(
            "unknown device '{name}'; expected one of {ALL_NAMES:?}"
        )));
    }
    let device = construct(name).ok_or_else(|| {
        Error::value(format!(
            "device '{name}' was not compiled into this build; available devices: {:?}",
            available_devices()
        ))
    })?;
    if name != "cpu" {
        if let Err(reason) = probe_detail(&device) {
            return Err(Error::os(format!(
                "device '{name}' is compiled in but not usable on this machine: {reason}"
            )));
        }
    }
    Ok(device)
}

/// The canonical name for a `DispatchDevice`, as returned by `PointCloud.device`.
pub fn device_name(device: &DispatchDevice) -> &'static str {
    match device {
        DispatchDevice::Flex(_) => "cpu",
        #[cfg(target_os = "macos")]
        DispatchDevice::Metal(_) => "metal",
        #[cfg(not(target_os = "macos"))]
        DispatchDevice::Vulkan(_) => "vulkan",
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(_) => "cuda",
        #[cfg(feature = "torch")]
        DispatchDevice::LibTorch(_) => "torch",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}

/// Whether coordinates on `device` can be held as plain CPU-resident f64
/// (ADR-0002). True for the pure-Rust CPU backend, and for LibTorch only on
/// its CPU device, whose tensors live in host memory and support f64. False
/// for wgpu (Metal/Vulkan) and CUDA devices, and for LibTorch's own MPS,
/// CUDA, and Vulkan devices: those have no f64 path and need the f32-relative
/// representation.
pub fn is_cpu_device(device: &DispatchDevice) -> bool {
    match device {
        DispatchDevice::Flex(_) => true,
        #[cfg(feature = "torch")]
        DispatchDevice::LibTorch(device) => {
            matches!(device, burn::backend::libtorch::LibTorchDevice::Cpu)
        }
        #[allow(unreachable_patterns)]
        _ => false,
    }
}
