use super::*;

use std::path::PathBuf;

fn get_extraction_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../test-corpus/content/fixtures")
        .join(name)
}

async fn assert_extracts_text(
    file_name: &str,
    expected_text: &str,
) -> Result<(), FileExtractionError> {
    let file_path = get_extraction_fixture(file_name);

    let extraction = document::from_path(
        &file_path,
        &ExtractionConfig {
            return_content: true,
            ..Default::default()
        },
    )
    .await?;
    let Some(text) = extraction.text else {
        panic!("File {} did not return text", file_name);
    };
    let preview = text.chars().take(100).collect::<String>();

    assert!(
        text.contains(expected_text),
        "File {} does not contain expected text: {}",
        file_name,
        preview
    );

    Ok(())
}

macro_rules! extract_from_files {
    ($expected_content:expr; $($test_name:ident => $file_name:expr),+ $(,)?) => {
        $(
        #[tokio::test]
        async fn $test_name() -> Result<(), FileExtractionError> {
            assert_extracts_text($file_name, $expected_content).await
        }
        )+
    };
}

macro_rules! unsupported_format_test {
    ($test_name:ident, $file_name:expr) => {
        #[tokio::test]
        async fn $test_name() -> Result<(), FileExtractionError> {
            let file_path = get_extraction_fixture($file_name);
            let error = match document::from_path(
                &file_path,
                &ExtractionConfig {
                    return_content: true,
                    ..Default::default()
                },
            )
            .await
            {
                Ok(_) => {
                    panic!("File {} should be unsupported", $file_name)
                }
                Err(error) => error,
            };

            assert!(
                matches!(
                    error,
                    FileExtractionError::UnsupportedFileType { .. }
                ),
                "File {} returned unexpected error: {}",
                $file_name,
                error
            );

            Ok(())
        }
    };
}

const SAMPLE_TEXT: &str = "life is but an instant; his substance";
const SAMPLE_CODE: &str = "extraction fixture marker: shared code sample";

// Supported document formats
extract_from_files!(SAMPLE_TEXT;
    supported_doc => "text.doc",
    supported_docx => "text.docx",
    supported_epub => "text.epub",
    supported_md => "text.md",
    supported_pdf => "text.pdf",
    supported_pptx => "text.pptx",
    supported_rss => "text.rss",
    supported_rtf => "text.rtf",
    supported_txt => "text.txt",
    supported_xhtml => "text.xhtml",
    supported_xml => "text.xml",
);

// Supported code formats
extract_from_files!(SAMPLE_CODE;
    supported_c => "code.c",
    supported_cpp => "code.cpp",
    supported_cs => "code.cs",
    supported_css => "code.css",
    supported_go => "code.go",
    supported_html => "code.html",
    supported_java => "code.java",
    supported_js => "code.js",
    supported_php => "code.php",
    supported_py => "code.py",
    supported_rb => "code.rb",
    supported_rs => "code.rs",
    supported_sh => "code.sh",
    supported_sql => "code.sql",
    supported_toml => "code.toml",
    supported_ts => "code.ts",
    supported_yaml => "code.yaml",
);

// Currently known faulty formats
unsupported_format_test!(unsupported_webp, "image-excel-table.webp");
unsupported_format_test!(unsupported_odt, "text.odt");
unsupported_format_test!(unsupported_zip, "text.txt.zip");

#[tokio::test]
async fn extracts_text_with_metadata() -> Result<(), FileExtractionError> {
    let file_path = get_extraction_fixture("text.txt");
    let extraction = document::from_path(
        &file_path,
        &ExtractionConfig {
            return_content: true,
            ..Default::default()
        },
    )
    .await?;
    let Some(metadata) = extraction.metadata else {
        panic!("File text.txt did not return metadata");
    };

    assert_eq!(metadata.extension.as_deref(), Some("txt"));
    assert_eq!(metadata.stem.as_deref(), Some("text"));
    assert!(extraction.text.is_some());

    Ok(())
}

#[tokio::test]
async fn extracts_metadata_without_text() -> Result<(), FileExtractionError> {
    let file_path = get_extraction_fixture("text.pptx");
    let extraction = document::from_path(
        &file_path,
        &ExtractionConfig {
            return_metadata: true,
            return_content: false,
            ..Default::default()
        },
    )
    .await?;
    let Some(metadata) = extraction.metadata else {
        panic!("File text.pptx did not return metadata");
    };

    assert_eq!(metadata.extension.as_deref(), Some("pptx"));
    assert!(extraction.text.is_none());

    Ok(())
}

#[tokio::test]
async fn returns_top_level_parts_without_text()
-> Result<(), FileExtractionError> {
    let file_path = get_extraction_fixture("text.txt");
    let extraction = document::from_path(
        &file_path,
        &ExtractionConfig {
            return_parts: true,
            ..Default::default()
        },
    )
    .await?;
    let parts = extraction.parts.expect("file should return parts");

    assert!(extraction.text.is_none());
    assert!(!parts.is_empty());
    assert!(parts.iter().any(|part| {
        part.text
            .as_deref()
            .is_some_and(|text| text.contains(SAMPLE_TEXT))
    }));

    Ok(())
}

#[tokio::test]
async fn returns_parts_for_syntax_text_fixtures()
-> Result<(), FileExtractionError> {
    for file_name in [
        "text.epub",
        "code.html",
        "text.rss",
        "text.xhtml",
        "text.xml",
    ] {
        let file_path = get_extraction_fixture(file_name);
        let extraction = document::from_path(
            &file_path,
            &ExtractionConfig {
                return_parts: true,
                ..Default::default()
            },
        )
        .await?;
        let parts = extraction.parts.expect("file should return parts");

        assert!(!parts.is_empty(), "{file_name} should produce parts");
    }

    Ok(())
}

#[cfg(feature = "ocr")]
#[test]
fn extracts_png_with_ocr() {
    let handle = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let runtime = tokio::runtime::Runtime::new()
                .expect("tokio runtime should start");
            runtime.block_on(async {
                let file_path = get_extraction_fixture("text-hidpi.png");
                let extraction = document::from_path(
                    &file_path,
                    &ExtractionConfig {
                        return_content: true,
                        return_parts: true,
                        ..Default::default()
                    },
                )
                .await
                .expect("OCR extraction should succeed");

                assert!(
                    extraction
                        .text
                        .is_some_and(|text| text.contains("On Looking Inward"))
                );
                let parts = extraction.parts.expect("OCR should return parts");
                assert!(parts.len() > 1);
                assert!(parts.iter().all(|part| part.kind == PartKind::Text));
            });
        })
        .expect("OCR test thread should start");

    handle.join().expect("OCR test thread should finish");
}
