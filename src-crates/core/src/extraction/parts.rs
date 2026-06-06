//! Shared extraction part helpers.

use crate::extraction::{ExtractionPart, PartKind};

/// Build simple text parts from non-empty lines.
pub(in crate::extraction) fn from_text(text: &str) -> Vec<ExtractionPart> {
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
