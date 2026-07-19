use std::collections::BTreeMap;
use std::path::Path;

use crate::extraction::{
    DocumentContent, ExtractionBbox, ExtractionPart, ExtractionPipelineStep,
    ExtractionPipelineStepKind, FileExtractionError, PartKind, pipeline,
    provenance,
};

/// Extract OCR parts from image files.
pub(in crate::extraction) async fn extract(
    file_path: &Path,
    ocr_options: &crate::ocr::OcrEngineOptions,
) -> Result<DocumentContent, FileExtractionError> {
    let ocr = crate::ocr::OcrEngine::new(ocr_options.clone())
        .await
        .map_err(ocr_extraction_error)?;
    let pipeline_config = ocr.pipeline();

    let started = std::time::Instant::now();
    let page = ocr.extract_file(file_path).map_err(ocr_extraction_error)?;
    let duration_ms = started.elapsed().as_millis() as u64;

    let block_count = page.blocks.len();
    let pipeline = vec![pipeline::step(
        ExtractionPipelineStepKind::Recognition,
        pipeline_config.recognition_model.to_string(),
        duration_ms,
        BTreeMap::from([("texts".to_owned(), block_count as u64)]),
    )];

    Ok(from_ocr_page(&page, pipeline))
}

/// Build extraction parts from OCR page output.
fn from_ocr_page(
    page: &crate::ocr::OcrPage,
    pipeline: Vec<ExtractionPipelineStep>,
) -> DocumentContent {
    let parts = page
        .blocks
        .iter()
        .filter_map(|block| {
            let text = block.text.trim();
            (!text.is_empty()).then_some((block, text))
        })
        .enumerate()
        .map(|(index, (block, text))| ExtractionPart {
            index,
            kind: PartKind::Text,
            text: Some(text.to_owned()),
            provenance: Some(provenance::from_page_bbox(
                None,
                ExtractionBbox {
                    x: block.bbox.x,
                    y: block.bbox.y,
                    width: block.bbox.width,
                    height: block.bbox.height,
                },
                block.confidence,
            )),
        })
        .collect::<Vec<_>>();

    DocumentContent {
        canonical_text: None,
        parts,
        pipeline,
    }
}

/// Map OCR errors into extraction engine errors.
fn ocr_extraction_error(source: crate::ocr::OcrError) -> FileExtractionError {
    FileExtractionError::ExtractionEngine {
        engine: "ocr",
        source: Box::new(source),
    }
}
