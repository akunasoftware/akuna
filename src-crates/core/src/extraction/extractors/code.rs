use std::collections::BTreeMap;

use crate::extraction::{
    DocumentContent, ExtractionMetadata, ExtractionPipelineStepKind, parts,
    pipeline,
};

/// Extract structured syntax parts from text content when language is supported.
pub(in crate::extraction) fn extract(
    text: &str,
    metadata: &ExtractionMetadata,
) -> Option<DocumentContent> {
    let extension = metadata.extension.as_deref();
    if !crate::chunking::code::supports_code_extension(extension) {
        return None;
    }

    let started = std::time::Instant::now();
    let parts = crate::chunking::code::segment_code(text, extension)
        .map(parts::from_segments)
        .unwrap_or_else(|| parts::from_text(text));

    let part_count = parts.len();
    let duration_ms = started.elapsed().as_millis() as u64;
    Some(DocumentContent {
        canonical_text: Some(text.to_owned()),
        parts,
        pipeline: vec![pipeline::step(
            ExtractionPipelineStepKind::Parsing,
            "tree-sitter",
            duration_ms,
            BTreeMap::from([("parts".to_owned(), part_count as u64)]),
        )],
    })
}
