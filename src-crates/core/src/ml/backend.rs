//! Shared ML backend selection.

use std::sync::OnceLock;

use burn_dispatch::{Dispatch, DispatchDevice};
use burn_ndarray::NdArrayDevice;
use burn_wgpu::WgpuDevice;

/// Shared ML backend.
pub(crate) type Backend = Dispatch;

/// A CPU device.
pub(crate) fn cpu_device() -> DispatchDevice {
    DispatchDevice::NdArray(NdArrayDevice::default())
}

/// A GPU device.
pub(crate) fn gpu_device() -> DispatchDevice {
    DispatchDevice::Wgpu(WgpuDevice::default())
}

/// Default ML device.
pub(crate) fn active_device() -> DispatchDevice {
    if gpu_available() {
        gpu_device()
    } else {
        cpu_device()
    }
}

/// Whether a usable GPU is present.
pub(crate) fn gpu_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(probe_gpu)
}

fn probe_gpu() -> bool {
    // Probe the single graphics API cubecl-wgpu's `AutoGraphicsApi` selects
    // (Metal on macOS, Vulkan elsewhere), so we never report an adapter for a
    // backend cubecl would not actually use.
    let backends = if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    let request = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    });
    pollster::block_on(request).is_ok()
}
