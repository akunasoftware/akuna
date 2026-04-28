use std::path::Path;

use pdf_oxide::converters::ConversionOptions;

use crate::{
    TextExtractionConfig,
    types::extraction::{DetectedFileType, FileExtractionError},
};

const EXPLICIT_UNSUPPORTED_MIMES: &[&str] = &[
    "application/zip",                         // .zip
    "application/vnd.oasis.opendocument.text", // .odt
];

/// Extracts text (markdown when possible) from most plain or rich file types at given path
/// requires input of magika info detection
pub(super) async fn extract_text(
    config: Option<&TextExtractionConfig>,
    file_path: &Path,
    detected_type: &DetectedFileType,
) -> Result<String, FileExtractionError> {
    let unsupported_error = FileExtractionError::UnsupportedFileType {
        mime_type: detected_type.mime_type.clone(),
        label: detected_type.label.clone(),
        description: detected_type.description.clone(),
    };

    if EXPLICIT_UNSUPPORTED_MIMES.contains(&detected_type.mime_type.as_str()) {
        return Err(unsupported_error);
    }

    // handle explicit type extractions first
    let content: Option<String> = match detected_type.mime_type.as_str() {

        // pdf_oxide
        "application/pdf"  // .pdf
        => Some(extract_pdf(config, file_path)?),

        // office_oxide
        "application/msword" |  // .doc
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" | // .pptx
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" // .docx
        => {
            Some(extract_office(config, file_path)?)
        }

        // rbook
        "application/epub+zip" //.epub
        => Some(extract_epub(file_path)?),

        // fall-through and allow generic is_text to handle rest
        _ => None,
    };

    // return now if got something from explicit parsers
    if let Some(content) = content {
        return Ok(content);
    }

    // basic text fallback
    if detected_type.is_text {
        return extract_multi(config, detected_type, file_path).await;
    }

    // make no more attempts, do not just loop-try with generic parsers
    // throw explicit failure and come back to improve lib + add test case
    Err(unsupported_error)
}

/// Extracts pdf files
fn extract_pdf(
    config: Option<&TextExtractionConfig>,
    file_path: &Path,
) -> Result<String, FileExtractionError> {
    let doc = pdf_oxide::PdfDocument::open(file_path)?;
    let content = if config.is_some_and(|c| c.prefer_markdown) {
        doc.to_markdown_all(&ConversionOptions {
            ..Default::default()
        })?
    } else {
        doc.extract_all_text()?
    };
    Ok(content)
}

/// Extracts office documents (word, ppt, etc.)
fn extract_office(
    config: Option<&TextExtractionConfig>,
    file_path: &Path,
) -> Result<String, FileExtractionError> {
    let doc = office_oxide::Document::open(file_path)?;
    let content = if config.is_some_and(|c| c.prefer_markdown) {
        doc.to_markdown()
    } else {
        doc.plain_text()
    };
    Ok(content)
}

/// Extracts epub to html+xml, then that to plaintext
fn extract_epub(file_path: &Path) -> Result<String, FileExtractionError> {
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

/// Extracts multiple file types using our inferred file type info
async fn extract_multi(
    _config: Option<&TextExtractionConfig>,
    file_type: &DetectedFileType,
    file_path: &Path,
) -> Result<String, FileExtractionError> {
    let preferred_mime = preferred_omniparse_mime(file_type);

    let bytes = tokio::fs::read(file_path).await?;
    let doc = omniparse::extract_from_bytes(&bytes, Some(preferred_mime))?;

    if let omniparse::Content::Text(text) = doc.content {
        return Ok(text);
    }

    Err(FileExtractionError::MissingTextContent {
        engine: "omniparse",
    })
}

/// Returns the best MIME hint for omniparse from detected file info.
fn preferred_omniparse_mime(file_type: &DetectedFileType) -> &str {
    match file_type.mime_type.as_str() {
        "text/rtf" => "application/rtf",
        mime_type if omniparse::is_mime_supported(mime_type) => mime_type,
        _ if file_type.is_text => "text/plain",
        _ => file_type.mime_type.as_str(),
    }
}

impl From<burn_magika::Error> for FileExtractionError {
    fn from(source: burn_magika::Error) -> Self {
        Self::ExtractionEngine {
            engine: "magika",
            source: Box::new(source),
        }
    }
}

impl From<pdf_oxide::Error> for FileExtractionError {
    fn from(source: pdf_oxide::Error) -> Self {
        Self::ExtractionEngine {
            engine: "pdf_oxide",
            source: Box::new(source),
        }
    }
}

impl From<office_oxide::OfficeError> for FileExtractionError {
    fn from(source: office_oxide::OfficeError) -> Self {
        Self::ExtractionEngine {
            engine: "office_oxide",
            source: Box::new(source),
        }
    }
}

impl From<omniparse::Error> for FileExtractionError {
    fn from(source: omniparse::Error) -> Self {
        Self::ExtractionEngine {
            engine: "omniparse",
            source: Box::new(source),
        }
    }
}

impl From<rbook::ebook::errors::EbookError> for FileExtractionError {
    fn from(source: rbook::ebook::errors::EbookError) -> Self {
        Self::ExtractionEngine {
            engine: "rbook-epub",
            source: Box::new(source),
        }
    }
}

impl From<rbook::ebook::errors::ArchiveError> for FileExtractionError {
    fn from(source: rbook::ebook::errors::ArchiveError) -> Self {
        Self::ExtractionEngine {
            engine: "rbook-epub",
            source: Box::new(source),
        }
    }
}

impl From<rbook::reader::errors::ReaderError> for FileExtractionError {
    fn from(source: rbook::reader::errors::ReaderError) -> Self {
        Self::ExtractionEngine {
            engine: "rbook-epub",
            source: Box::new(source),
        }
    }
}
