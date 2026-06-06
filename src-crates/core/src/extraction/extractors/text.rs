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
        let doc =
            omniparse::extract_from_bytes(html.as_bytes(), Some("text/html"))?;

        if let omniparse::Content::Text(chapter_text) = doc.content {
            let chapter_text = chapter_text.trim();
            if chapter_text.is_empty() {
                continue;
            }

            text.push_str(chapter_text);
            text.push_str("\n\n");
        }
    }

    Ok(text.trim().to_string())
}

/// Extract text bytes via omniparse using detected file info as MIME hint.
pub(in crate::extraction) fn extract_bytes_with_omniparse(
    metadata: &ExtractionMetadata,
    bytes: &[u8],
) -> Result<String, FileExtractionError> {
    let preferred_mime = preferred_omniparse_mime(metadata);
    let doc = omniparse::extract_from_bytes(bytes, Some(preferred_mime))?;

    if let omniparse::Content::Text(text) = doc.content {
        return Ok(text);
    }

    Err(FileExtractionError::MissingTextContent {
        engine: "omniparse",
    })
}

/// Returns best MIME hint for omniparse from detected file info.
fn preferred_omniparse_mime(metadata: &ExtractionMetadata) -> &str {
    match metadata.mime_type.as_str() {
        "text/rtf" => "application/rtf",
        mime_type if omniparse::is_mime_supported(mime_type) => mime_type,
        _ if metadata.is_text => "text/plain",
        _ => metadata.mime_type.as_str(),
    }
}
