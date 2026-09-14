use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use burn_dispatch::DispatchDevice;

use crate::detection::models::magika::MagikaModel;
use crate::detection::vendor::model as vendor_model;
#[cfg(any(feature = "extraction", test))]
use crate::detection::vendor::{content, file::TypeInfo as VendorTypeInfo};
use crate::detection::{DetectionError, FileType};
use crate::ml::backend::{self, Backend};

#[cfg(test)]
mod tests;

static DEFAULT_MODEL: Mutex<Option<Arc<MagikaModel<Backend>>>> =
    Mutex::new(None);

// Files not well supported by Magika.
#[cfg(any(feature = "extraction", test))]
const NIX: VendorTypeInfo = VendorTypeInfo {
    label: "nix",
    mime_type: "text/x-nix",
    group: "code",
    description: "Nix source",
    extensions: &["nix"],
    is_text: true,
};

/// Detects file types from bytes and files.
pub struct FileTypeDetector {
    model: Arc<MagikaModel<Backend>>,
}

impl FileTypeDetector {
    /// Builds a detector sharing the process-wide CPU model tuned for single-file detection.
    pub fn new() -> Result<Self, DetectionError> {
        // CPU is faster for single-file inference; explicit devices bypass this
        // cache. Only construction is locked, never inference.
        let mut model =
            DEFAULT_MODEL.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(model) = model.as_ref() {
            return Ok(Self {
                model: Arc::clone(model),
            });
        }

        // Publish only a complete model so errors and panics can be retried.
        let detector = Self::new_on(backend::cpu_device())?;
        *model = Some(Arc::clone(&detector.model));
        Ok(detector)
    }

    /// Builds a detector on a specific device.
    pub(crate) fn new_on(
        device: DispatchDevice,
    ) -> Result<Self, DetectionError> {
        let model = MagikaModel::<Backend>::from_embedded(&device)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Identifies the file type of raw bytes (blocking).
    pub fn identify_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<FileType, DetectionError> {
        self.model.identify_bytes(bytes)
    }

    /// Identifies valid UTF-8 from a known filename extension.
    #[cfg(any(feature = "extraction", test))]
    pub(crate) fn identify_utf8_extension(
        extension: &str,
        bytes: &[u8],
    ) -> Option<FileType> {
        let extension = extension.trim_start_matches('.');
        let lowercase = extension
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
            .then(|| extension.to_ascii_lowercase());
        let info = match lowercase.as_deref().unwrap_or(extension) {
            // File types worth fast-path skipping Magika.
            "css" => &content::CSS,
            "csv" => &content::CSV,
            "go" => &content::GO,
            "ini" => &content::INI,
            "java" => &content::JAVA,
            "js" => &content::JAVASCRIPT,
            "json" => &content::JSON,
            "md" => &content::MARKDOWN,
            "php" => &content::PHP,
            "py" => &content::PYTHON,
            "rb" => &content::RUBY,
            "rs" => &content::RUST,
            "sh" => &content::SHELL,
            "sql" => &content::SQL,
            "toml" => &content::TOML,
            "ts" => &content::TYPESCRIPT,
            "tsv" => &content::TSV,
            "txt" => &content::TXT,
            "yaml" | "yml" => &content::YAML,
            "nix" => &NIX,
            _ => return None,
        };
        std::str::from_utf8(bytes).ok()?;
        Some(FileType::ruled_info(info))
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
    let path = path.as_ref();
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "detection input is not a regular file",
        )
        .into());
    }

    let mut file = File::open(path)?;
    let block_size = vendor_model::CONFIG.block_size;
    let sample_size = block_size * 2;
    let mut bytes = Vec::with_capacity(sample_size);

    if metadata.len() <= sample_size as u64 {
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
