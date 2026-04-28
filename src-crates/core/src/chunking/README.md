# Chunking Namespace

`akuna_core::chunking` chunks text content using configured delimiters and target size.
Feature-gated behind `chunking`.

Use `chunk_text` directly when content is already available.

## How To Use It

Call `chunk_text` with optional `ChunkingConfig`, text content, and an optional file extension.
Without config, default chunk sizing and delimiter behaviour are used.
Use `delimiters_by_ft` when a file type needs its own boundary rules.
Use `delimiters` as the fallback boundary rule for all other content.

| Input            | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| `config`         | Optional target size and delimiter configuration.     |
| `content`        | Source text to split.                                 |
| `file_extension` | Extension used to pick file-type-specific delimiters. |

## Full Configuration

```rust
use std::collections::HashMap;

use akuna_core::{ChunkingConfig, chunking::chunk_text};

let mut delimiters_by_ft = HashMap::new();
delimiters_by_ft.insert("md".to_string(), b"\n\n".to_vec());

let config = ChunkingConfig {
    target_size: Some(512),
    delimiters_by_ft,
    delimiters: Some(b"\n".to_vec()),
};

let chunks = chunk_text(Some(&config), "hello\nworld", Some("md"));
```
