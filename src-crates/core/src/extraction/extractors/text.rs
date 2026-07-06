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
        text.push_str(&plain_text_from_markup(&html));
        text.push('\n');
    }

    Ok(text.trim().to_string())
}

/// Extract plain text from bytes.
pub(in crate::extraction) fn extract_bytes(
    metadata: &ExtractionMetadata,
    bytes: &[u8],
) -> Result<String, FileExtractionError> {
    let text = std::str::from_utf8(bytes).map_err(|source| {
        FileExtractionError::ExtractionEngine {
            engine: "direct",
            source: Box::new(source),
        }
    })?;

    if is_markup(&metadata.mime_type) {
        return Ok(plain_text_from_markup(text));
    }

    Ok(text.to_string())
}

/// Returns whether the MIME type contains markup text.
fn is_markup(mime_type: &str) -> bool {
    mime_type.ends_with("+xml")
        || matches!(
            mime_type,
            "application/rss+xml"
                | "application/xhtml+xml"
                | "application/xml"
                | "text/html"
                | "text/xml"
        )
}

/// Converts simple tag-based markup to normalized visible text.
fn plain_text_from_markup(markup: &str) -> String {
    let mut text = String::with_capacity(markup.len());
    let mut in_tag = false;
    let mut quote = None;
    let mut last_was_whitespace = true;

    for character in markup.chars() {
        match character {
            '<' if !in_tag => {
                in_tag = true;
                quote = None;
                push_space(&mut text, &mut last_was_whitespace);
            }
            '"' | '\'' if in_tag => {
                if quote == Some(character) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(character);
                }
            }
            '>' if in_tag && quote.is_none() => in_tag = false,
            _ if in_tag => {}
            _ if character.is_whitespace() => {
                push_space(&mut text, &mut last_was_whitespace);
            }
            _ => {
                text.push(character);
                last_was_whitespace = false;
            }
        }
    }

    decode_common_entities(text.trim()).trim().to_string()
}

/// Appends one whitespace separator when needed.
fn push_space(text: &mut String, last_was_whitespace: &mut bool) {
    if !*last_was_whitespace {
        text.push(' ');
        *last_was_whitespace = true;
    }
}

/// Decodes entities common in fixture markup.
fn decode_common_entities(text: &str) -> String {
    text.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}
