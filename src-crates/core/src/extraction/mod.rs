use std::path::Path;

mod content;
mod metadata;

use content::extract_text;
use metadata::extract_metadata;

use crate::chunking::chunk_text;
use crate::{
    ExtractionChunk, ExtractionConfig, ExtractionContent, ExtractionResult,
};

pub use crate::types::extraction::FileExtractionError;

/// Configurable main entrypoint to extract content & metadata from a file on disk
pub async fn extract_file(
    file_path: impl AsRef<Path>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult, FileExtractionError> {
    let file_path = file_path.as_ref();
    validate_file(file_path)?;

    let need_metadata = config.return_metadata
        || config.return_content
        || config.return_chunking;
    let need_content = config.return_content || config.return_chunking;
    let need_chunks = config.return_chunking;

    let file_inference = if need_metadata {
        Some(extract_metadata(file_path).await?)
    } else {
        None
    };

    let content = if let (Some(file_inference), true) =
        (file_inference.as_ref(), need_content)
    {
        Some(
            extract_text(
                config.text.as_ref(),
                file_path,
                &file_inference.detected_type,
            )
            .await?,
        )
    } else {
        None
    };

    let chunk_texts = if let (Some(file_inference), Some(content), true) =
        (file_inference.as_ref(), content.as_ref(), need_chunks)
    {
        Some(chunk_text(
            config.chunking.as_ref(),
            content,
            file_inference.metadata.extension.as_deref(),
        ))
    } else {
        None
    };

    let metadata = file_inference.and_then(|extraction| {
        config.return_metadata.then_some(extraction.metadata)
    });
    let chunks = chunk_texts.and_then(|chunk_texts| {
        config.return_chunking.then(|| {
            chunk_texts
                .into_iter()
                .enumerate()
                .map(|(index, text)| ExtractionChunk {
                    index,
                    text: Some(text.to_owned()),
                })
                .collect()
        })
    });
    let content = config
        .return_content
        .then_some(ExtractionContent { text: content });

    Ok(ExtractionResult {
        metadata,
        content,
        chunks,
    })
}

/// Sanity check that given path is actually a real file
fn validate_file(file_path: &Path) -> Result<(), FileExtractionError> {
    // confirm file exists
    if !file_path.exists() {
        return Err(FileExtractionError::Io {
            source: std::io::Error::other("Given path does not exist"),
        });
    }

    // confirm file is readable
    if !file_path.is_file() {
        return Err(FileExtractionError::Io {
            source: std::io::Error::other("Given path is not a file"),
        });
    }

    // confirm file has bytes
    if file_path.metadata()?.len() == 0 {
        return Err(FileExtractionError::NoContents);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::ExtractionConfig;
    use crate::extraction::*;
    use crate::testing::get_extraction_fixture;

    async fn assert_extracts_content(
        file_name: &str,
        expected_content: &str,
    ) -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture(file_name);

        let extraction = extract_file(
            &file_path,
            &ExtractionConfig {
                return_content: true,
                ..Default::default()
            },
        )
        .await?;
        let Some(content) = extraction.content else {
            panic!("File {} did not return content", file_name);
        };
        let Some(text) = content.text else {
            panic!("File {} did not return content text", file_name);
        };
        let preview = text.chars().take(100).collect::<String>();

        assert!(
            text.contains(expected_content),
            "File {} does not contain expected text in content: {}",
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
                assert_extracts_content($file_name, $expected_content).await
            }
            )+
        };
    }

    macro_rules! unsupported_format_test {
        ($test_name:ident, $file_name:expr) => {
            #[tokio::test]
            async fn $test_name() -> Result<(), FileExtractionError> {
                let file_path = get_extraction_fixture($file_name);
                let error = match extract_file(
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
    async fn extracts_content_with_metadata() -> Result<(), FileExtractionError>
    {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_content: true,
                return_chunking: true,
                ..Default::default()
            },
        )
        .await?;
        let Some(metadata) = extraction.metadata else {
            panic!("File text.txt did not return metadata");
        };

        assert_eq!(metadata.extension.as_deref(), Some("txt"));
        assert_eq!(metadata.stem.as_deref(), Some("text"));
        assert!(extraction.content.is_some());
        assert!(extraction.chunks.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn extracts_metadata_without_content()
    -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.pptx");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_metadata: true,
                return_content: false,
                return_chunking: false,
                ..Default::default()
            },
        )
        .await?;
        let Some(metadata) = extraction.metadata else {
            panic!("File text.pptx did not return metadata");
        };

        assert_eq!(metadata.extension.as_deref(), Some("pptx"));
        assert!(extraction.content.is_none());
        assert!(extraction.chunks.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn extracts_chunks_without_content() -> Result<(), FileExtractionError>
    {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_content: false,
                return_chunking: true,
                ..Default::default()
            },
        )
        .await?;

        assert!(extraction.content.is_none());
        assert!(extraction.chunks.is_some_and(|chunks| !chunks.is_empty()));

        Ok(())
    }
}
