use std::path::Path;

use quick_xml::events::Event;
use quick_xml::reader::Reader;

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
        text.push_str(
            &plain_text_from_markup(&html)
                .map_err(|source| markup_error("rbook-epub", source))?,
        );
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
        return plain_text_from_markup(text)
            .map_err(|source| markup_error("direct", source));
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

/// Converts tag-based markup to normalized visible text.
fn plain_text_from_markup(markup: &str) -> quick_xml::Result<String> {
    let mut text = String::with_capacity(markup.len());
    let mut last_was_whitespace = true;
    let mut remaining = markup;

    'fragments: loop {
        let mut reader = Reader::from_str(remaining);
        reader.config_mut().allow_dangling_amp = true;
        reader.config_mut().allow_unmatched_ends = true;
        reader.config_mut().check_end_names = false;

        loop {
            match reader.read_event()? {
                Event::Start(element)
                    if is_hidden_element(element.local_name().as_ref()) =>
                {
                    push_space(&mut text, &mut last_was_whitespace);
                    let name = element.name();
                    let Some(rest) = after_hidden_element(
                        remaining,
                        reader.buffer_position() as usize,
                        name.as_ref(),
                    ) else {
                        break 'fragments;
                    };
                    remaining = rest;
                    continue 'fragments;
                }
                Event::Start(_) | Event::Empty(_) | Event::End(_) => {
                    push_space(&mut text, &mut last_was_whitespace);
                }
                Event::Text(value) => {
                    push_text(
                        &mut text,
                        &mut last_was_whitespace,
                        &value.html_content()?,
                    );
                }
                Event::CData(value) => {
                    push_text(
                        &mut text,
                        &mut last_was_whitespace,
                        &value.decode()?,
                    );
                }
                Event::GeneralRef(value) => {
                    let value =
                        if let Some(character) = value.resolve_char_ref()? {
                            character.to_string()
                        } else {
                            let name = value.decode()?;
                            match name.as_ref() {
                                "amp" => "&".to_string(),
                                "apos" => "'".to_string(),
                                "gt" => ">".to_string(),
                                "lt" => "<".to_string(),
                                "nbsp" => " ".to_string(),
                                "quot" => "\"".to_string(),
                                _ => format!("&{name};"),
                            }
                        };
                    push_text(&mut text, &mut last_was_whitespace, &value);
                }
                Event::Eof => break 'fragments,
                _ => {}
            }
        }
    }

    Ok(text.trim().to_string())
}

/// Returns whether an element's content is not visible document text.
fn is_hidden_element(name: &[u8]) -> bool {
    name.eq_ignore_ascii_case(b"script") || name.eq_ignore_ascii_case(b"style")
}

/// Returns the markup after a raw-text element's closing tag.
fn after_hidden_element<'a>(
    markup: &'a str,
    start: usize,
    name: &[u8],
) -> Option<&'a str> {
    let bytes = markup.as_bytes();
    let close_start = bytes[start..]
        .iter()
        .enumerate()
        .filter(|(_, byte)| **byte == b'<')
        .map(|(offset, _)| start + offset)
        .find(|tag_start| {
            bytes.get(tag_start + 1) == Some(&b'/')
                && tag_name_matches(bytes, tag_start + 2, name)
        })?;
    let close_end = markup[close_start..].find('>')? + close_start;
    Some(&markup[close_end + 1..])
}

/// Matches an ASCII tag name followed by a valid boundary.
fn tag_name_matches(bytes: &[u8], start: usize, name: &[u8]) -> bool {
    let end = start + name.len();
    bytes
        .get(start..end)
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(name))
        && bytes.get(end).is_none_or(|byte| {
            byte.is_ascii_whitespace() || matches!(byte, b'/' | b'>')
        })
}

/// Appends text while collapsing whitespace.
fn push_text(text: &mut String, last_was_whitespace: &mut bool, value: &str) {
    for character in value.chars() {
        if character.is_whitespace() {
            push_space(text, last_was_whitespace);
        } else {
            text.push(character);
            *last_was_whitespace = false;
        }
    }
}

/// Appends one whitespace separator when needed.
fn push_space(text: &mut String, last_was_whitespace: &mut bool) {
    if !*last_was_whitespace {
        text.push(' ');
        *last_was_whitespace = true;
    }
}

/// Converts parser failures at the extraction boundary.
fn markup_error(
    engine: &'static str,
    source: quick_xml::Error,
) -> FileExtractionError {
    FileExtractionError::ExtractionEngine {
        engine,
        source: Box::new(source),
    }
}
