//! Shared extraction provenance helpers.

use crate::extraction::{
    ExtractionBbox, ExtractionByteRange, ExtractionProvenance,
};

/// Build provenance for byte-local source text.
pub(in crate::extraction) fn from_byte_range(
    start: usize,
    end: usize,
) -> ExtractionProvenance {
    ExtractionProvenance {
        confidence: None,
        page: None,
        bbox: None,
        byte_range: Some(ExtractionByteRange { start, end }),
    }
}

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
        byte_range: None,
    }
}
