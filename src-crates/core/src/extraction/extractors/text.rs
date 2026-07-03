use std::path::Path;

use crate::extraction::{ExtractionMetadata, FileExtractionError};

/// Extract EPUB content by rendering each chapter to plain text.
pub(in crate::extraction) fn extract_epub(
    file_path: &Path,
) -> Result<String, FileExtractionError> {
    let doc = rbook::Epub::open(file_path)?;

    let mut text = String::new();
    for data_result in doc.reader() {
        let data = data_result?;
        let html = data.into_string();
        text = html
    }

    Ok(text.trim().to_string())
}

/// Extract text bytes via omniparse using detected file info as MIME hint.
pub(in crate::extraction) fn extract_bytes(
    _metadata: &ExtractionMetadata,
    bytes: &[u8],
) -> Result<String, FileExtractionError> {
    // TODO: More sophisticated file-type parsing for text file types

    let text = std::str::from_utf8(bytes).map_err(|source| {
        FileExtractionError::ExtractionEngine {
            engine: "direct",
            source: Box::new(source),
        }
    })?;

    Ok(text.to_string())
}
