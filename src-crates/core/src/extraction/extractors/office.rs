use std::collections::BTreeMap;
use std::path::Path;

use crate::extraction::{
    DocumentContent, ExtractionPipelineStepKind, FileExtractionError, pipeline,
};

/// Extract structured text and parts from Office documents.
pub(in crate::extraction) fn extract(
    file_path: &Path,
) -> Result<DocumentContent, FileExtractionError> {
    let started = std::time::Instant::now();
    let document = office_oxide::Document::open(file_path)?;
    let mut content = DocumentContent::from_text(document.plain_text());
    let duration_ms = started.elapsed().as_millis() as u64;
    let audit = pipeline::step(
        ExtractionPipelineStepKind::Parsing,
        "office_oxide",
        duration_ms,
        BTreeMap::from([("parts".to_owned(), content.parts.len() as u64)]),
    );

    content.pipeline.push(audit);
    Ok(content)
}
