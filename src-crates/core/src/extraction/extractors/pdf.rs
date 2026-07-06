use std::collections::BTreeMap;
use std::path::Path;

use crate::extraction::{
    DocumentContent, ExtractionBbox, ExtractionPart, FileExtractionError,
    PartKind, pipeline, provenance,
};

#[derive(Clone)]
struct PdfPart {
    kind: PartKind,
    text: String,
    page: usize,
    bbox: (f32, f32, f32, f32),
}

impl PdfPart {
    /// Convert PDF-local part into public extraction part.
    fn into_extraction_part(self, index: usize) -> ExtractionPart {
        ExtractionPart {
            index,
            kind: self.kind,
            text: Some(self.text),
            provenance: Some(provenance::from_page_bbox(
                Some(self.page),
                ExtractionBbox {
                    x: self.bbox.0,
                    y: self.bbox.1,
                    width: self.bbox.2,
                    height: self.bbox.3,
                },
                None,
            )),
        }
    }
}

/// Extract structured text and parts from PDF documents.
pub(in crate::extraction) fn extract(
    file_path: &Path,
) -> Result<DocumentContent, FileExtractionError> {
    use pdf_oxide::extractors::{DocumentElement, StructuredExtractor};

    let started = std::time::Instant::now();
    let mut document = pdf_oxide::PdfDocument::open(file_path)?;
    let page_count = document.page_count()?;
    let mut extractor = StructuredExtractor::new();
    let mut parts = Vec::new();

    for page_index in 0..page_count {
        let structured =
            extractor.extract_page(&mut document, page_index as u32)?;
        for element in structured.elements {
            let element_parts = match element {
                DocumentElement::Header { text, bbox, .. } => {
                    vec![(PartKind::Heading, text, bbox)]
                }
                DocumentElement::Paragraph { text, bbox, .. } => {
                    vec![(PartKind::Paragraph, text, bbox)]
                }
                DocumentElement::List { items, bbox, .. } => items
                    .into_iter()
                    .map(|item| item.text)
                    .map(|text| (PartKind::ListItem, text, bbox))
                    .collect::<Vec<_>>(),
                DocumentElement::Table { cells, bbox, .. } => cells
                    .into_iter()
                    .flatten()
                    .map(|text| (PartKind::Table, text, bbox))
                    .collect::<Vec<_>>(),
            };

            for (kind, text, bbox) in element_parts {
                let text = text.trim();
                if text.is_empty() {
                    continue;
                }

                parts.push(PdfPart {
                    kind,
                    text: text.to_owned(),
                    page: page_index + 1,
                    bbox,
                });
            }
        }
    }

    let text = document.extract_all_text()?;

    let parts = parts
        .into_iter()
        .enumerate()
        .map(|(index, part)| part.into_extraction_part(index))
        .collect::<Vec<_>>();
    let part_count = parts.len();
    let audit = pipeline::step(
        crate::extraction::ExtractionPipelineStepKind::Parsing,
        "pdf_oxide",
        started.elapsed().as_millis() as u64,
        BTreeMap::from([
            ("pages".to_owned(), page_count as u64),
            ("parts".to_owned(), part_count as u64),
        ]),
    );

    if !parts.is_empty() {
        return Ok(DocumentContent {
            canonical_text: Some(text),
            parts,
            pipeline: vec![audit],
        });
    }

    let mut content = DocumentContent::from_text(text);
    content.pipeline.push(audit);
    Ok(content)
}
