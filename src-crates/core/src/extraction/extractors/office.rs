use std::collections::HashMap;
use std::path::Path;

use crate::extraction::FileExtractionError;
use crate::extraction::{DocumentContent, ExtractionPart, PartKind, pipeline};

/// Extract structured text and parts from Office documents.
pub(in crate::extraction) fn extract(
    file_path: &Path,
) -> Result<DocumentContent, FileExtractionError> {
    let started = std::time::Instant::now();
    let document =
        office_oxide::Document::open(file_path).map_err(|error| {
            FileExtractionError::extraction_engine("office_oxide", error)
        })?;
    let text = document.plain_text();
    let parts = structured_parts(&text);
    let duration_ms = started.elapsed().as_millis() as u64;

    if parts.len() > 1 {
        let part_count = parts.len();
        return Ok(DocumentContent {
            canonical_text: None,
            parts,
            pipeline: vec![pipeline::step(
                "parsing",
                "office_oxide",
                duration_ms,
                HashMap::from([("parts".to_owned(), part_count)]),
            )],
        });
    }

    Ok(DocumentContent::from_text(text))
}

/// Build structured parts from Office plain text output.
fn structured_parts(text: &str) -> Vec<ExtractionPart> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(index, line)| ExtractionPart {
            index,
            kind: PartKind::Text,
            text: Some(line.to_owned()),
            provenance: None,
        })
        .collect()
}
