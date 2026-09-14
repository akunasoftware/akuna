//! Document extraction entry points.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::fs::File;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::extraction::{
    DocumentContent, ExtractionConfig, ExtractionMetadata,
    ExtractionPipelineStepKind, ExtractionResult, FileExtractionError,
    extractors, metadata, pipeline,
};

/// Extract content and metadata from a document path.
pub async fn extract_file(
    file_path: &Path,
    config: &ExtractionConfig,
) -> Result<ExtractionResult, FileExtractionError> {
    validate_path(file_path)?;
    let bytes = tokio::fs::read(file_path).await?;

    from_bytes_with_source_path(&bytes, Some(file_path), config).await
}

/// Extract content and metadata from encoded document bytes.
pub async fn extract_bytes(
    bytes: &[u8],
    config: &ExtractionConfig,
) -> Result<ExtractionResult, FileExtractionError> {
    from_bytes_with_source_path(bytes, None, config).await
}

// Shared extraction path for both owned bytes and files.
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

    let (metadata, mut pipeline) = if need_metadata {
        let started = Instant::now();
        let metadata = metadata::from_bytes(bytes, source_path)?;
        let duration_ms = started.elapsed().as_millis() as u64;
        let step = pipeline::step(
            ExtractionPipelineStepKind::Detection,
            "magika",
            duration_ms,
            BTreeMap::from([("types".to_owned(), 1)]),
        );
        (Some(metadata), vec![step])
    } else {
        (None, Vec::new())
    };

    let content =
        if let (Some(metadata), true) = (metadata.as_ref(), need_content) {
            Some(extract_content(bytes, source_path, metadata, config).await?)
        } else {
            None
        };

    let (returned_text, parts) = if let Some(content) = content {
        let text = config.return_content.then(|| content.text()).flatten();
        pipeline.extend(content.pipeline);
        (text, config.return_parts.then_some(content.parts))
    } else {
        (None, None)
    };
    let returned_metadata = if config.return_metadata {
        metadata
    } else {
        None
    };
    Ok(ExtractionResult {
        metadata: returned_metadata,
        pipeline,
        text: returned_text,
        parts,
    })
}

// Dispatch by detected type.
async fn extract_content(
    bytes: &[u8],
    source_path: Option<&Path>,
    metadata: &ExtractionMetadata,
    _config: &ExtractionConfig,
) -> Result<DocumentContent, FileExtractionError> {
    let started = Instant::now();
    let (mut content, engine) = match metadata.mime_type.as_str() {
        // PDF documents use structural page extraction.
        "application/pdf" => {
            let temporary_file =
                temporary_content_file(bytes, source_path, metadata).await?;
            return extractors::pdf::extract_file(&temporary_file.path);
        }

        // Office documents use document-level structural extraction.
        "application/msword"
        | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        | "application/vnd.openxmlformats-officedocument.wordprocessingml.document" =>
        {
            let temporary_file =
                temporary_content_file(bytes, source_path, metadata).await?;
            return extractors::office::extract_file(&temporary_file.path);
        }

        // EPUB needs chapter rendering before text normalization.
        "application/epub+zip" => {
            let temporary_file =
                temporary_content_file(bytes, source_path, metadata).await?;
            (
                extractors::text::extract_epub_file(&temporary_file.path)
                    .map(DocumentContent::from_text)?,
                "rbook-epub",
            )
        }

        // Markup/text container formats need text normalization.
        "application/rss+xml"
        | "application/xhtml+xml"
        | "application/xml"
        | "text/html"
        | "text/markdown"
        | "text/xml" => (
            extractors::text::extract_bytes(metadata, bytes)
                .map(DocumentContent::from_text)?,
            "direct",
        ),

        mime_type if mime_type.ends_with("+xml") => (
            extractors::text::extract_bytes(metadata, bytes)
                .map(DocumentContent::from_text)?,
            "direct",
        ),

        // Known archive formats are not treated as text fallbacks.
        "application/zip" | "application/vnd.oasis.opendocument.text" => {
            return Err(unsupported_file_type(metadata));
        }

        // Images use OCR when the feature is enabled.
        #[cfg(feature = "ocr")]
        "image/bmp" | "image/jpeg" | "image/png" | "image/tiff" => {
            return extractors::ocr::extract_bytes(bytes, &_config.ocr).await;
        }

        // Remaining detected text goes through generic parser fallback.
        _ if metadata.is_text => (
            extractors::text::extract_bytes(metadata, bytes)
                .map(DocumentContent::from_text)?,
            "direct",
        ),

        // Everything else needs an explicit extractor first.
        _ => return Err(unsupported_file_type(metadata)),
    };
    content.pipeline.insert(
        0,
        pipeline::step(
            ExtractionPipelineStepKind::Parsing,
            engine,
            started.elapsed().as_millis() as u64,
            BTreeMap::from([
                ("parts".to_owned(), content.parts.len() as u64),
                ("texts".to_owned(), u64::from(content.text().is_some())),
            ]),
        ),
    );
    Ok(content)
}

// Some extractors require a path.
pub(super) async fn temporary_content_file(
    bytes: &[u8],
    source_path: Option<&Path>,
    metadata: &ExtractionMetadata,
) -> Result<TemporaryFileCleanup, FileExtractionError> {
    for attempt in 0..100_u8 {
        let file_path = temporary_content_path(source_path, metadata, attempt);
        // Async open could create a file after cancellation, before guarding it.
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&file_path);
        let file = match file {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                continue;
            }
            Err(error) => return Err(error.into()),
        };

        let cleanup = TemporaryFileCleanup::new(file_path);
        return cleanup.write(File::from_std(file), bytes).await;
    }

    Err(FileExtractionError::Io {
        source: Error::new(
            ErrorKind::AlreadyExists,
            "could not create temporary extraction file",
        ),
    })
}

// Removes temporary files when path-backed extraction finishes.
pub(super) struct TemporaryFileCleanup {
    pub(super) path: PathBuf,
}

impl TemporaryFileCleanup {
    // Track temporary file for cleanup at scope exit.
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(super) async fn write(
        self,
        file: impl AsyncWrite + Unpin,
        bytes: &[u8],
    ) -> Result<Self, FileExtractionError> {
        // Drop the writer before the guard on errors and cancellation, too.
        let mut file = file;
        file.write_all(bytes).await?;
        file.flush().await?;
        drop(file);
        Ok(self)
    }
}

impl Drop for TemporaryFileCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// Build a temporary content path.
fn temporary_content_path(
    source_path: Option<&Path>,
    metadata: &ExtractionMetadata,
    attempt: u8,
) -> PathBuf {
    let extension = source_path
        .and_then(Path::extension)
        .map(|extension| extension.to_string_lossy().to_string())
        .or_else(|| metadata.extension.clone())
        .or_else(|| {
            extension_from_mime(&metadata.mime_type).map(str::to_string)
        });
    let file_name = format!(
        "extraction-{}-{}-{}{}",
        std::process::id(),
        unique_suffix(),
        attempt,
        extension
            .as_ref()
            .map(|extension| format!(".{extension}"))
            .unwrap_or_default()
    );

    std::env::temp_dir().join(file_name)
}

// Returns an extension for path-based extractors when input bytes have no path.
fn extension_from_mime(mime_type: &str) -> Option<&'static str> {
    match mime_type {
        "application/pdf" => Some("pdf"),
        "application/msword" => Some("doc"),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => {
            Some("pptx")
        }
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some("docx")
        }
        "application/epub+zip" => Some("epub"),
        _ => None,
    }
}

// Best-effort unique suffix for temporary file names.
fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

// Validate file input before extraction.
fn validate_path(file_path: &Path) -> Result<(), FileExtractionError> {
    if !file_path.exists() {
        return Err(FileExtractionError::Io {
            source: Error::other(format!(
                "Given path does not exist: {}",
                file_path.to_string_lossy()
            )),
        });
    }

    if !file_path.is_file() {
        return Err(FileExtractionError::Io {
            source: Error::other(format!(
                "Given path is not a file: {}",
                file_path.to_string_lossy()
            )),
        });
    }

    if file_path.metadata()?.len() == 0 {
        return Err(FileExtractionError::NoContents);
    }

    Ok(())
}

// Preserve detected metadata on unsupported type errors.
fn unsupported_file_type(metadata: &ExtractionMetadata) -> FileExtractionError {
    FileExtractionError::UnsupportedFileType {
        metadata: Box::new(metadata.clone()),
    }
}
