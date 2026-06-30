//! Runtime backend and device selection for the library.

use std::sync::OnceLock;

use burn_dispatch::{Dispatch, DispatchDevice};
use burn_ndarray::NdArrayDevice;
use burn_wgpu::WgpuDevice;

/// The single backend the whole library runs on.
pub(crate) type Backend = Dispatch;

/// A CPU device.
pub(crate) fn cpu_device() -> DispatchDevice {
    DispatchDevice::NdArray(NdArrayDevice::default())
}

/// A GPU device.
pub(crate) fn gpu_device() -> DispatchDevice {
    DispatchDevice::Wgpu(WgpuDevice::default())
}

/// The device the library uses by default: the GPU when available, else the CPU.
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

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::{Backend, cpu_device, gpu_available};

    /// The CPU device must run real tensor ops, since it is the fallback used on
    /// machines without a GPU (e.g. CI).
    #[test]
    fn cpu_backend_runs_matmul() {
        let device = cpu_device();
        let a = Tensor::<Backend, 2>::from_floats(
            [[1.0, 2.0], [3.0, 4.0]],
            &device,
        );
        let b = Tensor::<Backend, 2>::from_floats(
            [[5.0, 6.0], [7.0, 8.0]],
            &device,
        );
        let out = a
            .matmul(b)
            .into_data()
            .to_vec::<f32>()
            .expect("matmul output to f32");
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    /// The probe must not panic and must return a definite, stable answer.
    #[test]
    fn gpu_probe_is_stable() {
        assert_eq!(gpu_available(), gpu_available());
    }
}
