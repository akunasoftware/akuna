//! Shared extraction part helpers.

use crate::chunking::Segment;
use crate::extraction::{ExtractionPart, provenance};

/// Build prose parts from text.
pub(in crate::extraction) fn from_text(text: &str) -> Vec<ExtractionPart> {
    from_segments(crate::chunking::prose::segment_prose(text))
}

/// Build extraction parts from source segments.
pub(in crate::extraction) fn from_segments(
    segments: Vec<Segment<'_>>,
) -> Vec<ExtractionPart> {
    segments
        .into_iter()
        .filter_map(|segment| {
            (!segment.text.trim().is_empty()).then(|| {
                (segment.kind, segment.byte_range, segment.text.to_owned())
            })
        })
        .enumerate()
        .map(|(index, (kind, byte_range, text))| ExtractionPart {
            index,
            kind,
            text: Some(text),
            provenance: Some(provenance::from_byte_range(
                byte_range.start,
                byte_range.end,
            )),
        })
        .collect()
}
