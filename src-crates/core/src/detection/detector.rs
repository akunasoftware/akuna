use std::path::Path;

use burn_dispatch::DispatchDevice;

use crate::detection::models::magika::MagikaModel;
use crate::detection::{FileType, MagikaInferenceError};
use crate::ml::backend::{self, Backend};

/// Detects file types from bytes and files.
pub struct FileTypeDetector {
    model: MagikaModel<Backend>,
}

impl FileTypeDetector {
    /// Builds a detector on the default device.
    pub fn new() -> Result<Self, MagikaInferenceError> {
        Self::new_on(backend::active_device())
    }

    /// Builds a detector on a specific device.
    pub(crate) fn new_on(
        device: DispatchDevice,
    ) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<Backend>::from_embedded(&device)?;
        Ok(Self { model })
    }

    /// Identifies the file type of raw bytes (blocking).
    pub fn identify_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_bytes(bytes)
    }

    /// Identifies the file type of a file path (blocking).
    pub fn identify_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<FileType, MagikaInferenceError> {
        let bytes = std::fs::read(path).map_err(MagikaInferenceError::Io)?;
        self.identify_bytes(&bytes)
    }
}
