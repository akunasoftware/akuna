use burn_dispatch::DispatchDevice;

use crate::detection::models::magika::MagikaModel;
use crate::detection::{FileType, MagikaInferenceError};
use crate::ml::backend::{self, Backend};

/// A file-type detection session ready to classify inputs.
pub struct Session {
    model: MagikaModel<Backend>,
}

impl Session {
    /// Builds a session on the default device.
    pub fn new() -> Result<Self, MagikaInferenceError> {
        Self::new_on(backend::active_device())
    }

    /// Builds a session on a specific device.
    pub(crate) fn new_on(
        device: DispatchDevice,
    ) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<Backend>::from_embedded(&device)?;
        Ok(Self { model })
    }

    /// Identifies the file type of raw bytes (blocking).
    pub fn identify_content_sync(
        &self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_bytes(bytes)
    }
}
