//! Metadata extraction entry points.

use std::path::Path;

use crate::extraction::{ExtractionMetadata, FileExtractionError};

/// Detect the file type and assemble metadata from raw bytes.
pub(in crate::extraction) fn from_bytes(
    bytes: &[u8],
    source_path: Option<&Path>,
) -> Result<ExtractionMetadata, FileExtractionError> {
    let detected = detect_content_type(bytes)?;
    let extension = source_path.and_then(|path| {
        path.extension()
            .map(|extension| extension.to_string_lossy().into_owned())
    });
    let stem = source_path.and_then(|path| {
        path.file_stem()
            .map(|file_stem| file_stem.to_string_lossy().into_owned())
    });

    Ok(ExtractionMetadata {
        stem,
        extension,
        label: detected.label,
        mime_type: detected.mime_type,
        description: detected.description,
        is_text: detected.is_text,
        hash: blake3::hash(bytes).to_hex().to_string(),
    })
}

/// Detection result used internally before assembling full metadata.
struct DetectionResult {
    mime_type: String,
    label: String,
    description: String,
    is_text: bool,
}

/// Detect the file type from bytes.
fn detect_content_type(
    bytes: &[u8],
) -> Result<DetectionResult, FileExtractionError> {
    let magika = crate::detection::Session::new()?;
    let type_info = magika.identify_content_sync(bytes)?.info();

    Ok(DetectionResult {
        mime_type: type_info.mime_type.to_string(),
        label: type_info.label.to_string(),
        description: type_info.description.to_string(),
        is_text: type_info.is_text,
    })
}
