# Extraction Namespace

`akuna_core::extraction` extracts file metadata, text content, and text chunks.
Feature-gated behind `extraction`.

Use `extract_file` for the high-level extraction flow.
The default config returns metadata only.

## How To Use It

Configure `ExtractionConfig` around the output you want back.
Metadata is cheap and default.
Content extraction reads the file body and uses detected file type to choose an extractor.
Chunking derives chunks from extracted content and can return chunks without returning full content.

| Option                 | Effect                                                    |
| ---------------------- | --------------------------------------------------------- |
| `return_metadata`      | Return inferred file metadata and detected type.          |
| `return_content`       | Return extracted text content.                            |
| `return_chunking`      | Return text chunks derived from extracted content.        |
| `text.prefer_markdown` | Prefer Markdown output where extractors support it.       |
| `chunking`             | Configure target size and delimiters for returned chunks. |

## Full Configuration

```rust
use std::collections::HashMap;

use akuna_core::{
    ChunkingConfig, ExtractionConfig, TextExtractionConfig,
    extraction::extract_file,
};

let mut delimiters_by_ft = HashMap::new();
delimiters_by_ft.insert("md".to_string(), b"\n\n".to_vec());

let config = ExtractionConfig {
    return_metadata: true,
    return_content: true,
    return_chunking: true,
    text: Some(TextExtractionConfig {
        prefer_markdown: true,
    }),
    chunking: Some(ChunkingConfig {
        target_size: Some(512),
        delimiters_by_ft,
        delimiters: Some(b"\n".to_vec()),
    }),
};

let result = extract_file("./notes.md", &config).await?;
```
