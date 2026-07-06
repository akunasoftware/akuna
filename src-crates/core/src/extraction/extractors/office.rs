use std::collections::HashMap;
use std::path::Path;

use crate::extraction::FileExtractionError;
use crate::extraction::{
    DocumentContent, ExtractionPipelineStepKind, parts::from_text, pipeline,
};

/// Extract structured text and parts from Office documents.
pub(in crate::extraction) fn extract(
    file_path: &Path,
) -> Result<DocumentContent, FileExtractionError> {
    let started = std::time::Instant::now();
    let document = office_oxide::Document::open(file_path)?;
    let text = document.plain_text();
    let parts = from_text(&text);
    let duration_ms = started.elapsed().as_millis() as u64;

    if parts.len() > 1 {
        let part_count = parts.len();
        return Ok(DocumentContent {
            canonical_text: None,
            parts,
            pipeline: vec![pipeline::step(
                ExtractionPipelineStepKind::Parsing,
                "office_oxide",
                duration_ms,
                HashMap::from([("parts".to_owned(), part_count as u64)]),
            )],
        });
    }

    Ok(DocumentContent::from_text(text))
}
