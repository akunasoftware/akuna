//! Shared extraction provenance helpers.

use crate::extraction::{ExtractionBbox, ExtractionProvenance};

/// Build provenance for page-local bounding boxes.
pub(in crate::extraction) fn from_page_bbox(
    page: Option<usize>,
    bbox: ExtractionBbox,
    confidence: Option<f32>,
) -> ExtractionProvenance {
    ExtractionProvenance {
        confidence,
        page,
        bbox: Some(bbox),
    }
}
