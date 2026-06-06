# File Extraction

Extracts text, structured parts, and metadata from files on disk.
File type is detected via Magika and routed to PDF, office document, EPUB, or
generic text extraction.

## Usage

```rust
use akuna_core::extraction::{extract_file, ExtractionConfig};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExtractionConfig {
        return_metadata: true,
        return_content: true,
        ..ExtractionConfig::default()
    };
    let result = extract_file("path/to/file.pdf", &config).await?;
    println!("{}", result.text.unwrap());
    Ok(())
}
```

## Configuration

`ExtractionConfig` controls which outputs are produced:

| Field             | Effect                                              |
| ----------------- | --------------------------------------------------- |
| `return_metadata` | Include inferred file metadata in the result.       |
| `return_content`  | Include text derived from parts in the result.      |
| `return_parts`    | Include structured content parts in the result.     |

Metadata inference does not read file contents.
Content is only read when `return_content` or `return_parts` is enabled.

## Supported Formats

- PDF (`.pdf`) via `pdf_oxide`
- Word (`.doc`, `.docx`) and PowerPoint (`.pptx`) via `office_oxide`
- EPUB (`.epub`) via `rbook`
- Markdown, RTF, RSS, XHTML, XML, plain text via `omniparse`
- Source code: Bash, C, C++, C#, CSS, Dart, Go, GraphQL, Groovy, HCL/Terraform, HTML, Java, JavaScript, JSON, Lua, PHP, Python, Ruby, Rust, Scala, SCSS, SQL, Svelte, Swift, TOML, TypeScript, Vue, YAML

Unsupported types return `FileExtractionError::UnsupportedFileType`.
