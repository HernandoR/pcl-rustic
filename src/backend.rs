//! Runtime device selection for the unified `burn::Dispatch` backend
//! (ADR-0001).
//!
//! Feature mapping (see Cargo.toml):
//!
//! | crate feature | burn feature | `DispatchDevice` variant |
//! |---|---|---|
//! | `cpu`    | `burn/flex`   | `Flex` (pure-Rust CPU backend) |
//! | `vulkan` | `burn/vulkan` | `Vulkan` (wgpu/Vulkan) |
//! | `cuda`   | `burn/cuda`   | `Cuda` |
//! | `torch`  | `burn/tch`    | `LibTorch` |
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
use std::sync::LazyLock;

pub use burn::DispatchDevice;

use crate::error::{Error, Result};

/// The single tensor backend used throughout the crate (ADR-0001).
pub type B = Dispatch;

/// Attempts to construct the compiled `DispatchDevice` for a given name.
/// Returns `None` when the corresponding cargo feature was not compiled in.
fn construct(name: &str) -> Option<DispatchDevice> {
    match name {
        #[cfg(feature = "cpu")]
        "cpu" => Some(DispatchDevice::Flex(Default::default())),
        #[cfg(feature = "vulkan")]
        "vulkan" => Some(DispatchDevice::Vulkan(Default::default())),
        #[cfg(feature = "cuda")]
        "cuda" => Some(DispatchDevice::Cuda(Default::default())),
        #[cfg(feature = "torch")]
        "torch" => Some(DispatchDevice::LibTorch(Default::default())),
        _ => None,
    }
}

/// Device names in GPU-first preference order, used by `default_device`.
/// `cpu` is intentionally last: it is the universal fallback.
const PREFERENCE_ORDER: &[&str] = &["cuda", "vulkan", "torch", "cpu"];

/// Every device name this build knows how to construct, regardless of
/// whether it was actually compiled in (`construct` filters that).
const ALL_NAMES: &[&str] = &["cpu", "vulkan", "cuda", "torch"];

/// Attempts a trivial tensor round-trip on `device`, catching the panics
/// that GPU backends raise when no matching adapter is present at runtime.
/// This is deliberately distinct from compile-time feature availability:
/// a Vulkan build on a headless CI box compiles fine but has no adapter.
fn probe(device: &DispatchDevice) -> bool {
    let device = device.clone();
    // GPU backends panic deep inside cubecl worker threads when no adapter
    // exists; every thread's panic goes through the global hook, so silence
    // it for the probe window to keep headless-host fallback quiet.
    let saved_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let tensor = Tensor::<B, 1>::zeros([1], &device);
        let _ = tensor.into_data();
    }))
    .is_ok();
    std::panic::set_hook(saved_hook);
    ok
}

static DEFAULT_DEVICE: LazyLock<DispatchDevice> = LazyLock::new(|| {
    for name in PREFERENCE_ORDER {
        if let Some(device) = construct(name) {
            if *name == "cpu" || probe(&device) {
                return device;
            }
        }
    }
    // `cpu` is a default crate feature; if every candidate above
    // failed to construct (all features disabled) there is no
    // sensible fallback left, and this indicates a broken build.
    construct("cpu").expect("no dispatch backend feature was compiled in")
});

/// The default compute device: the first GPU variant compiled in whose
/// runtime probe succeeds, falling back to `cpu`. Probing failure degrades
/// gracefully -- never a panic at import (ADR-0001). Cached after first use.
pub fn default_device() -> DispatchDevice {
    DEFAULT_DEVICE.clone()
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

/// Parses a device name (`"cpu" | "vulkan" | "cuda" | "torch"`) into a
/// `DispatchDevice`, probing it (except `cpu`) so a caller asking for a
/// specific device gets a clear error instead of a later panic.
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
    if name != "cpu" && !probe(&device) {
        return Err(Error::os(format!(
            "device '{name}' is compiled in but not usable on this machine"
        )));
    }
    Ok(device)
}

/// The canonical name for a `DispatchDevice`, as returned by `PointCloud.device`.
pub fn device_name(device: &DispatchDevice) -> &'static str {
    match device {
        #[cfg(feature = "cpu")]
        DispatchDevice::Flex(_) => "cpu",
        #[cfg(feature = "vulkan")]
        DispatchDevice::Vulkan(_) => "vulkan",
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(_) => "cuda",
        #[cfg(feature = "torch")]
        DispatchDevice::LibTorch(_) => "torch",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}
