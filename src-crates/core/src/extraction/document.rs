//! Document extraction entry points.

use std::path::Path;
use std::path::PathBuf;

use tokio::io::AsyncWriteExt;

use crate::extraction::{
    DocumentContent, ExtractionConfig, ExtractionMetadata, ExtractionResult,
    FileExtractionError, extractors, metadata,
};

/// Extract content and metadata from a document path.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if path validation, type detection,
/// or content extraction fails.
pub async fn from_path(
    file_path: &Path,
    config: &ExtractionConfig,
) -> Result<ExtractionResult, FileExtractionError> {
    validate_path(file_path)?;
    let bytes = tokio::fs::read(file_path).await?;

    from_bytes_with_source_path(&bytes, Some(file_path), config).await
}

/// Extract content and metadata from bytes with optional source path.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if type detection or content extraction fails.
async fn from_bytes_with_source_path(
    bytes: &[u8],
    source_path: Option<&Path>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult, FileExtractionError> {
    if bytes.is_empty() {
        return Err(FileExtractionError::NoContents);
    }

    let need_metadata =
        config.return_metadata || config.return_content || config.return_parts;
    let need_content = config.return_content || config.return_parts;

    let metadata = if need_metadata {
        Some(metadata::from_bytes(bytes, source_path)?)
    } else {
        None
    };

    let content =
        if let (Some(metadata), true) = (metadata.as_ref(), need_content) {
            Some(extract_content(bytes, source_path, metadata, config).await?)
        } else {
            None
        };

    let text = content.as_ref().and_then(DocumentContent::text);
    let parts = content
        .as_ref()
        .and_then(|content| config.return_parts.then(|| content.parts.clone()));
    let returned_text = config.return_content.then_some(text).flatten();
    let returned_metadata = if config.return_metadata {
        metadata
    } else {
        None
    };
    let pipeline = content
        .as_ref()
        .map(|content| content.pipeline.clone())
        .unwrap_or_default();

    Ok(ExtractionResult {
        metadata: returned_metadata,
        pipeline,
        text: returned_text,
        parts,
    })
}

/// Build normalized internal document content from raw text.
fn content_from_text(
    text: impl Into<String>,
    metadata: &ExtractionMetadata,
) -> DocumentContent {
    let text = text.into();
    if let Some(content) = extractors::code::extract(&text, metadata) {
        return content;
    }

    DocumentContent::from_text(text)
}

/// Route detected document content to the right extractor.
async fn extract_content(
    bytes: &[u8],
    source_path: Option<&Path>,
    metadata: &ExtractionMetadata,
    config: &ExtractionConfig,
) -> Result<DocumentContent, FileExtractionError> {
    match metadata.mime_type.as_str() {
        // PDF documents use structural page extraction.
        "application/pdf" => {
            let file_path = temporary_content_file(bytes, source_path).await?;
            let _cleanup = TemporaryFileCleanup::new(file_path.clone());
            extractors::pdf::extract(&file_path)
        }

        // Office documents use document-level structural extraction.
        "application/msword"
        | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document" =>
        {
            let file_path = temporary_content_file(bytes, source_path).await?;
            let _cleanup = TemporaryFileCleanup::new(file_path.clone());
            extractors::office::extract(&file_path)
        }

        // EPUB needs chapter rendering before text normalization.
        "application/epub+zip" => {
            let file_path = temporary_content_file(bytes, source_path).await?;
            let _cleanup = TemporaryFileCleanup::new(file_path.clone());
            extractors::text::extract_epub(&file_path)
                .map(|text| content_from_text(text, metadata))
        }

        // Markup/text container formats use omniparse text extraction.
        "application/rss+xml"
        | "application/xhtml+xml"
        | "application/xml"
        | "text/html"
        | "text/markdown"
        | "text/xml" => {
            extractors::text::extract_bytes_with_omniparse(metadata, bytes)
                .map(|text| content_from_text(text, metadata))
        }

        // Known archive formats are not treated as text fallbacks.
        "application/zip" | "application/vnd.oasis.opendocument.text" => {
            Err(unsupported_file_type(metadata))
        }

        // Images use OCR when the feature is enabled.
        #[cfg(feature = "ocr")]
        "image/bmp" | "image/jpeg" | "image/png" | "image/tiff" => {
            let file_path = temporary_content_file(bytes, source_path).await?;
            let _cleanup = TemporaryFileCleanup::new(file_path.clone());
            extractors::ocr::extract(&file_path, &config.ocr).await
        }

        // Remaining detected text goes through generic parser fallback.
        _ if metadata.is_text => {
            extractors::text::extract_bytes_with_omniparse(metadata, bytes)
                .map(|text| content_from_text(text, metadata))
        }

        // Everything else needs an explicit extractor first.
        _ => Err(unsupported_file_type(metadata)),
    }
}

/// Write content bytes to temporary file for path-backed extractors.
async fn temporary_content_file(
    bytes: &[u8],
    source_path: Option<&Path>,
) -> Result<PathBuf, FileExtractionError> {
    for attempt in 0..100_u8 {
        let file_path = temporary_content_path(source_path, attempt);
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path)
            .await;
        let mut file = match file {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                continue;
            }
            Err(error) => return Err(error.into()),
        };

        file.write_all(bytes).await?;
        file.flush().await?;
        return Ok(file_path);
    }

    Err(FileExtractionError::Io {
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create temporary extraction file",
        ),
    })
}

/// Temporary path bridge for extractor libraries that require filesystem input.
struct TemporaryFileCleanup {
    path: PathBuf,
}

impl TemporaryFileCleanup {
    /// Track temporary file for cleanup at scope exit.
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for TemporaryFileCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Build unique temporary content path for byte-backed extraction.
fn temporary_content_path(source_path: Option<&Path>, attempt: u8) -> PathBuf {
    let extension = source_path
        .and_then(Path::extension)
        .map(|extension| extension.to_string_lossy());
    let file_name = format!(
        "extraction-{}-{}-{}{}",
        std::process::id(),
        unique_suffix(),
        attempt,
        extension
            .as_deref()
            .map(|extension| format!(".{extension}"))
            .unwrap_or_default()
    );

    std::env::temp_dir().join(file_name)
}

/// Return best-effort unique suffix for temporary file names.
fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

/// Confirm path exists, is readable file, and has non-zero size.
fn validate_path(file_path: &Path) -> Result<(), FileExtractionError> {
    if !file_path.exists() {
        return Err(FileExtractionError::Io {
            source: std::io::Error::other("Given path does not exist"),
        });
    }

    if !file_path.is_file() {
        return Err(FileExtractionError::Io {
            source: std::io::Error::other("Given path is not a file"),
        });
    }

    if file_path.metadata()?.len() == 0 {
        return Err(FileExtractionError::NoContents);
    }

    Ok(())
}

/// Build unsupported type error from detected metadata.
fn unsupported_file_type(metadata: &ExtractionMetadata) -> FileExtractionError {
    FileExtractionError::UnsupportedFileType {
        metadata: Box::new(metadata.clone()),
    }
}
