use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use burn_dispatch::DispatchDevice;

use crate::detection::models::magika::MagikaModel;
use crate::detection::vendor::model as vendor_model;
use crate::detection::{DetectionError, FileType};
use crate::ml::backend::{self, Backend};

/// Detects file types from bytes and files.
pub struct FileTypeDetector {
    model: MagikaModel<Backend>,
}

impl FileTypeDetector {
    /// Builds a detector on the default device.
    pub fn new() -> Result<Self, DetectionError> {
        Self::new_on(backend::active_device())
    }

    /// Builds a detector on a specific device.
    pub(crate) fn new_on(
        device: DispatchDevice,
    ) -> Result<Self, DetectionError> {
        let model = MagikaModel::<Backend>::from_embedded(&device)?;
        Ok(Self { model })
    }

    /// Identifies the file type of raw bytes (blocking).
    pub fn identify_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<FileType, DetectionError> {
        self.model.identify_bytes(bytes)
    }

    /// Identifies the file type of a file path (blocking).
    pub fn identify_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<FileType, DetectionError> {
        let bytes = read_file_sample(path)?;
        self.identify_bytes(&bytes)
    }
}

/// Reads the leading and trailing blocks used by Magika preprocessing.
pub(super) fn read_file_sample(
    path: impl AsRef<Path>,
) -> Result<Vec<u8>, DetectionError> {
    let mut file = File::open(path)?;
    let block_size = vendor_model::CONFIG.block_size;
    let sample_size = block_size * 2;
    let mut bytes = Vec::with_capacity(sample_size);

    if file.metadata()?.len() <= sample_size as u64 {
        file.take(sample_size as u64).read_to_end(&mut bytes)?;
        return Ok(bytes);
    }

    file.by_ref()
        .take(block_size as u64)
        .read_to_end(&mut bytes)?;
    file.seek(SeekFrom::End(-(block_size as i64)))?;
    file.take(block_size as u64).read_to_end(&mut bytes)?;
    Ok(bytes)
}
