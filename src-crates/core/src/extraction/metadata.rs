//! Metadata extraction entry points.

use std::path::Path;

use crate::detection::FileTypeDetector;
use crate::extraction::{ExtractionMetadata, FileExtractionError};

/// Detect the file type and assemble metadata from raw bytes.
pub(in crate::extraction) fn from_bytes(
    bytes: &[u8],
    source_path: Option<&Path>,
) -> Result<ExtractionMetadata, FileExtractionError> {
    let extension = source_path.and_then(|path| {
        path.extension()
            .map(|extension| extension.to_string_lossy().into_owned())
    });
    let detected = if let Some(detected) =
        extension.as_deref().and_then(|extension| {
            FileTypeDetector::identify_utf8_extension(extension, bytes)
        }) {
        detected
    } else {
        FileTypeDetector::new()?.identify_bytes(bytes)?
    };
    let info = detected.info();
    let stem = source_path.and_then(|path| {
        path.file_stem()
            .map(|file_stem| file_stem.to_string_lossy().into_owned())
    });

    Ok(ExtractionMetadata {
        stem,
        extension,
        label: info.label.clone(),
        mime_type: info.mime_type.clone(),
        description: info.description.clone(),
        is_text: info.is_text,
        confidence: detected.confidence(),
        origin: detected.origin(),
        hash: blake3::hash(bytes).to_hex().to_string(),
    })
}
