use std::future::{Future, poll_fn};
use std::path::PathBuf;
use std::task::Poll;

use tokio::fs::File;
use tokio::io::duplex;

use crate::extraction::document::temporary_content_file;
use crate::extraction::{
    DetectionOrigin, DocumentContent, ExtractionConfig, ExtractionMetadata,
    ExtractionPart, ExtractionPipelineStepKind, FileExtractionError, PartKind,
    extract_bytes, extract_file,
};

/// Fetches an extraction fixture from the shared corpus.
fn get_extraction_fixture(name: &str) -> PathBuf {
    crate::testkit::corpus_fixture(name).unwrap_or_else(|error| {
        panic!("Could not fetch {name} in corpus fixtures: {error:#}")
    })
}

const SAMPLE_TEXT: &str = "life is but an instant; his substance";
const SAMPLE_CODE: &str = "extraction fixture marker: shared code sample";

/// Builds test-only extraction metadata.
fn test_metadata(mime_type: &str) -> ExtractionMetadata {
    ExtractionMetadata {
        stem: None,
        extension: None,
        label: "test".to_string(),
        mime_type: mime_type.to_string(),
        description: "test".to_string(),
        is_text: true,
        confidence: 1.0,
        origin: DetectionOrigin::Rule,
        hash: "test".to_string(),
    }
}

#[tokio::test]
async fn cleans_temporary_file_after_success() -> Result<(), FileExtractionError>
{
    let bytes = b"temporary content";
    let temporary_file =
        temporary_content_file(bytes, None, &test_metadata("application/pdf"))
            .await?;
    let path = temporary_file.path.clone();

    assert_eq!(
        path.extension().and_then(|value| value.to_str()),
        Some("pdf")
    );
    assert_eq!(std::fs::read(&path)?, bytes);
    drop(temporary_file);
    assert!(!path.try_exists()?);
    Ok(())
}

#[tokio::test]
async fn cleans_temporary_file_after_write_failure()
-> Result<(), FileExtractionError> {
    let temporary_file =
        temporary_content_file(b"", None, &test_metadata("application/pdf"))
            .await?;
    let path = temporary_file.path.clone();
    let file = File::from_std(std::fs::File::open(&path)?);

    let result = temporary_file.write(file, b"cannot write").await;

    assert!(matches!(result, Err(FileExtractionError::Io { .. })));
    assert!(!path.try_exists()?);
    Ok(())
}

#[tokio::test]
async fn cleans_temporary_file_after_write_cancellation()
-> Result<(), FileExtractionError> {
    let temporary_file =
        temporary_content_file(b"", None, &test_metadata("application/pdf"))
            .await?;
    let path = temporary_file.path.clone();
    // A live, unread peer forces the second byte to remain pending.
    let (writer, _reader) = duplex(1);
    let mut writing = Box::pin(temporary_file.write(writer, b"ab"));

    assert!(
        poll_fn(|cx| Poll::Ready(writing.as_mut().poll(cx).is_pending())).await
    );
    assert!(path.try_exists()?);
    drop(writing);
    assert!(!path.try_exists()?);
    Ok(())
}

#[tokio::test]
async fn extracts_documents() -> Result<(), FileExtractionError> {
    for name in [
        "text.doc",
        "text.docx",
        "text.pptx",
        "text.pdf",
        "text.epub",
        "text.rss",
        "text.rtf",
        "text.xhtml",
        "text.xml",
    ] {
        let extraction = extract_file(
            &get_extraction_fixture(name),
            &ExtractionConfig {
                return_content: true,
                ..Default::default()
            },
        )
        .await?;
        assert!(
            extraction
                .text
                .as_deref()
                .is_some_and(|text| text.contains(SAMPLE_TEXT)),
            "{name} should contain document text"
        );
    }
    Ok(())
}

#[tokio::test]
async fn extracts_code_with_model_markup_and_rule()
-> Result<(), FileExtractionError> {
    for (file_name, origin) in [
        ("code.c", DetectionOrigin::Model),
        ("code.html", DetectionOrigin::Model),
        ("code.rs", DetectionOrigin::Rule),
    ] {
        let file_path = get_extraction_fixture(file_name);
        let extraction = extract_file(
            &file_path,
            &ExtractionConfig {
                return_content: true,
                ..Default::default()
            },
        )
        .await?;

        assert!(
            extraction
                .text
                .is_some_and(|text| text.contains(SAMPLE_CODE)),
            "{file_name} should contain the code sample"
        );
        let metadata =
            extraction.metadata.expect("metadata should be returned");
        assert_eq!(metadata.origin, origin, "{file_name}");
    }

    Ok(())
}

#[tokio::test]
async fn rejects_unsupported_formats() {
    for name in ["text.odt", "text.txt.zip"] {
        let error = extract_file(
            &get_extraction_fixture(name),
            &ExtractionConfig {
                return_content: true,
                ..Default::default()
            },
        )
        .await
        .expect_err("format should be unsupported");
        let FileExtractionError::UnsupportedFileType { metadata } = error
        else {
            panic!("{name} returned unexpected error: {error}");
        };
        if name.ends_with(".zip") {
            assert_eq!(metadata.mime_type, "application/zip");
        }
    }
}

#[tokio::test]
async fn extracts_text_with_metadata() -> Result<(), FileExtractionError> {
    let file_path = get_extraction_fixture("text.md");
    let extraction = extract_file(
        &file_path,
        &ExtractionConfig {
            return_content: true,
            ..Default::default()
        },
    )
    .await?;
    let Some(metadata) = extraction.metadata else {
        panic!("File text.md did not return metadata");
    };

    assert_eq!(metadata.extension.as_deref(), Some("md"));
    assert_eq!(metadata.stem.as_deref(), Some("text"));
    assert_eq!(metadata.label, "markdown");
    assert_eq!(metadata.origin, DetectionOrigin::Rule);
    assert!(
        extraction
            .text
            .as_deref()
            .is_some_and(|text| text.contains(SAMPLE_TEXT))
    );

    Ok(())
}

#[tokio::test]
async fn extracts_bytes_without_path_metadata()
-> Result<(), FileExtractionError> {
    let data = b"life is but an instant; his substance is fleeting\n";
    let extraction = extract_bytes(
        data,
        &ExtractionConfig {
            return_content: true,
            ..Default::default()
        },
    )
    .await?;
    let metadata = extraction.metadata.expect("bytes should return metadata");

    assert_eq!(metadata.extension, None);
    assert_eq!(metadata.stem, None);
    assert!(extraction.text.is_some_and(|text| text.contains("instant")));
    assert_eq!(
        extraction
            .pipeline
            .first()
            .expect("detection should be audited")
            .outputs
            .get("types"),
        Some(&1)
    );
    let direct = extraction
        .pipeline
        .iter()
        .find(|step| step.engine == "direct")
        .expect("direct parser should be audited");
    assert_eq!(direct.step, ExtractionPipelineStepKind::Parsing);
    assert_eq!(direct.outputs.get("parts"), Some(&1));
    assert_eq!(direct.outputs.get("texts"), Some(&1));

    Ok(())
}

#[test]
fn extracts_xml_suffix_bytes_as_text() -> Result<(), FileExtractionError> {
    let text = super::extractors::text::extract_bytes(
        &test_metadata("application/rdf+xml"),
        b"<rdf><label>life &amp; substance</label></rdf>",
    )?;

    assert_eq!(text, "life & substance");
    Ok(())
}

#[test]
fn extracts_markup_bytes_without_over_decoding_entities()
-> Result<(), FileExtractionError> {
    let text = super::extractors::text::extract_bytes(
        &test_metadata("text/html"),
        b"<p>&amp;lt;tag&amp;gt; &lt;real&gt;</p>",
    )?;

    assert_eq!(text, "&lt;tag&gt; <real>");
    Ok(())
}

#[test]
fn extracts_markup_bytes_with_quoted_delimiters()
-> Result<(), FileExtractionError> {
    let text = super::extractors::text::extract_bytes(
        &test_metadata("text/html"),
        b"<p title=\"a > b\">first</p><p title='c > d'>second</p>",
    )?;

    assert_eq!(text, "first second");
    Ok(())
}

#[test]
fn extracts_visible_markup_text() -> Result<(), FileExtractionError> {
    let text = super::extractors::text::extract_bytes(
        &test_metadata("text/html"),
        b"<style>.hidden::before { content: 'a < b'; }</style><p><![CDATA[visible <text>]]></p><SCRIPT type='text/javascript'>if (a < b) hidden()</SCRIPT>",
    )?;

    assert_eq!(text, "visible <text>");
    Ok(())
}

#[test]
fn trims_decoded_markup_entities() -> Result<(), FileExtractionError> {
    let text = super::extractors::text::extract_bytes(
        &test_metadata("text/html"),
        b"<p>&nbsp; text &nbsp;</p>",
    )?;

    assert_eq!(text, "text");
    Ok(())
}

#[tokio::test]
async fn extracts_office_bytes_without_path_extension()
-> Result<(), FileExtractionError> {
    let bytes = tokio::fs::read(get_extraction_fixture("text.docx")).await?;
    let extraction = extract_bytes(
        &bytes,
        &ExtractionConfig {
            return_content: true,
            ..Default::default()
        },
    )
    .await?;

    assert!(
        extraction
            .text
            .is_some_and(|text| text.contains(SAMPLE_TEXT))
    );
    assert_eq!(
        extraction
            .metadata
            .expect("bytes should return metadata")
            .origin,
        DetectionOrigin::Model
    );
    Ok(())
}

#[tokio::test]
async fn respects_output_flags() -> Result<(), FileExtractionError> {
    let file_path = get_extraction_fixture("text.txt");
    for (return_metadata, return_content, return_parts) in [
        (false, false, false),
        (true, false, false),
        (false, false, true),
        (false, true, false),
        (false, true, true),
    ] {
        let extraction = extract_file(
            &file_path,
            &ExtractionConfig {
                return_metadata,
                return_content,
                return_parts,
                ..Default::default()
            },
        )
        .await?;

        assert_eq!(extraction.metadata.is_some(), return_metadata);
        assert_eq!(extraction.text.is_some(), return_content);
        assert_eq!(extraction.parts.is_some(), return_parts);
        if let Some(text) = extraction.text {
            assert!(text.contains(SAMPLE_TEXT));
        }
        if let Some(parts) = extraction.parts {
            assert!(!parts.is_empty());
            assert!(parts.iter().any(|part| {
                part.text
                    .as_deref()
                    .is_some_and(|text| text.contains(SAMPLE_TEXT))
            }));
        }

        let expected_pipeline = if return_content || return_parts {
            vec![
                (ExtractionPipelineStepKind::Detection, "rule"),
                (ExtractionPipelineStepKind::Parsing, "direct"),
            ]
        } else if return_metadata {
            vec![(ExtractionPipelineStepKind::Detection, "rule")]
        } else {
            Vec::new()
        };
        assert_eq!(
            extraction
                .pipeline
                .iter()
                .map(|step| (step.step, step.engine.as_str()))
                .collect::<Vec<_>>(),
            expected_pipeline
        );
    }

    Ok(())
}

#[test]
fn preserves_plain_text_as_one_part() {
    let text = " first paragraph\n\nsecond paragraph ";
    let content = super::types::DocumentContent::from_text(text.to_string());

    assert_eq!(content.text().as_deref(), Some(text));
    assert_eq!(content.parts.len(), 1);
    assert_eq!(content.parts[0].kind, PartKind::Text);
    assert_eq!(content.parts[0].text.as_deref(), Some(text));
    assert!(content.parts[0].provenance.is_none());
}

#[test]
fn resolves_document_text() {
    for (canonical_text, parts, expected) in [
        (None, &[] as &[Option<&str>], None),
        (Some("canonical"), &[Some("part")], Some("canonical")),
        (Some(""), &[Some("part")], Some("")),
        (None, &[None, Some(""), None], None),
        (None, &[Some(""), None, Some("")], Some("\n\n")),
        (None, &[Some(" ")], Some(" ")),
        (
            None,
            &[Some(""), Some("text"), Some("")],
            Some("\n\ntext\n\n"),
        ),
        (
            None,
            &[Some("first"), Some("second")],
            Some("first\n\nsecond"),
        ),
    ] {
        let mut content = DocumentContent {
            canonical_text: canonical_text.map(str::to_owned),
            parts: parts
                .iter()
                .enumerate()
                .map(|(index, text)| ExtractionPart {
                    index,
                    kind: PartKind::Text,
                    text: text.map(str::to_owned),
                    provenance: None,
                })
                .collect(),
            pipeline: Vec::new(),
        };
        assert_eq!(content.has_text(), expected.is_some());
        assert_eq!(content.text().as_deref(), expected);
        assert_eq!(content.take_text().as_deref(), expected);
    }
}

#[test]
fn detection_io_errors_remain_io() {
    let error =
        FileExtractionError::from(crate::detection::DetectionError::Io {
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "missing test input",
            ),
        });

    assert!(matches!(
        error,
        FileExtractionError::Io { source }
            if source.kind() == std::io::ErrorKind::NotFound
    ));
}

#[cfg(feature = "ocr")]
#[test]
fn extracts_images_with_ocr() {
    crate::testkit::run_with_model_stack(|| {
        let runtime =
            tokio::runtime::Runtime::new().expect("tokio runtime should start");
        runtime.block_on(async {
            for name in [
                "text-hidpi.jpg",
                "text-hidpi.png",
                "text-hidpi.heic",
                "text-lowdpi.heic",
            ] {
                let file_path = get_extraction_fixture(name);
                let config = ExtractionConfig {
                    return_content: true,
                    return_parts: true,
                    ..Default::default()
                };
                let extraction = extract_file(&file_path, &config)
                    .await
                    .expect("OCR extraction should succeed");

                let text = extraction.text.expect("OCR should return text");
                assert!(text.contains("On Looking Inward"), "{name}");
                let parts = extraction.parts.expect("OCR should return parts");
                assert!(parts.len() > 1, "{name}");
                for (index, part) in parts.iter().enumerate() {
                    assert_eq!(part.index, index);
                    assert_eq!(part.kind, PartKind::Text);
                    let provenance = part
                        .provenance
                        .as_ref()
                        .expect("OCR part should have provenance");
                    assert!(provenance.bbox.is_some());
                    assert!(provenance.confidence.is_some());
                    assert_eq!(provenance.page, None);
                }
                assert_eq!(
                    text,
                    parts
                        .iter()
                        .map(|part| part
                            .text
                            .as_deref()
                            .expect("OCR part has text"))
                        .collect::<Vec<_>>()
                        .join("\n\n")
                );
                let recognition_engine =
                    config.ocr.recognition_model.to_string();
                // HEIC types resolve through the misdetection override rule.
                let detection_engine = if name.ends_with(".heic") {
                    "rule"
                } else {
                    "magika"
                };
                assert_eq!(
                    extraction
                        .pipeline
                        .iter()
                        .map(|step| (step.step, step.engine.as_str()))
                        .collect::<Vec<_>>(),
                    vec![
                        (
                            ExtractionPipelineStepKind::Detection,
                            detection_engine
                        ),
                        (
                            ExtractionPipelineStepKind::Recognition,
                            recognition_engine.as_str(),
                        ),
                    ]
                );
                assert_eq!(
                    extraction.pipeline[1].outputs.get("texts"),
                    Some(&(parts.len() as u64))
                );
            }
        });

        Ok(())
    })
    .expect("OCR test thread should finish");
}

#[cfg(feature = "ocr")]
#[test]
fn extracts_webp_with_ocr() {
    crate::testkit::run_with_model_stack(|| {
        let runtime =
            tokio::runtime::Runtime::new().expect("tokio runtime should start");
        runtime.block_on(async {
            for name in ["image-excel-barchart.webp", "image-excel-table.webp"]
            {
                let extraction = extract_file(
                    &get_extraction_fixture(name),
                    &ExtractionConfig {
                        return_content: true,
                        ..Default::default()
                    },
                )
                .await
                .expect("OCR extraction should succeed");
                let text = extraction.text.expect("OCR should return text");
                assert!(
                    !text.trim().is_empty(),
                    "{name} should OCR to some text"
                );
            }
        });

        Ok(())
    })
    .expect("OCR test thread should finish");
}
