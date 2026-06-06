use burn::tensor::backend::Backend;

use crate::detection::models::magika::MagikaModel;
use crate::detection::{Detection, FileType, MagikaInferenceError};

/// Default Burn backend used by the detection module.
pub(crate) type DefaultBackend = burn_wgpu::Wgpu;

/// High-level Magika inference session wrapping a Magika classifier.
pub struct Session<B: Backend> {
    model: MagikaModel<B>,
}

impl Session<DefaultBackend> {
    /// Builds a session on the default WGPU device.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded model cannot be loaded.
    pub fn new_default() -> Result<Self, MagikaInferenceError> {
        Self::new(&burn_wgpu::WgpuDevice::default())
    }
}

impl<B: Backend<FloatElem = f32>> Session<B> {
    /// Builds a session from the embedded model on the given device.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded model cannot be loaded.
    pub fn new(device: &B::Device) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<B>::from_embedded(device)?;
        Ok(Self { model })
    }

    /// Identifies the file type of raw bytes (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn identify_content_sync(
        &mut self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_bytes(bytes)
    }

    /// Classifies a batch of inputs and returns ranked alternatives each.
    ///
    /// # Errors
    ///
    /// Returns an error if any input fails inference.
    pub fn detect_content_batch_sync(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<Detection>, MagikaInferenceError> {
        self.model.detect_batch(inputs)
    }
}
