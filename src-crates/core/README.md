# akuna-core

Knowledge tooling library with feature-gated modules for detection, embedding,
extraction, OCR, reranking, and graph storage.

See the [workspace README](../../README.md) for the feature overview.

## Modules

| Module       | Feature      | Purpose                                |
| ------------ | ------------ | -------------------------------------- |
| `detection`  | `detection`  | File-type detection                    |
| `embedding`  | `embedding`  | Text embeddings                        |
| `extraction` | `extraction` | File extraction                        |
| `layout`     | `layout`     | Document layout detection              |
| `ocr`        | `ocr`        | Image text recognition                 |
| `reranking`  | `reranking`  | Text reranking                         |
| `storage`    | `storage`    | Graph storage and retrieval            |

The `full` feature enables every optional module above.

## Examples

### Extraction

```rust,no_run
use akuna_core::extraction::{extract_file, ExtractionConfig};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let result = extract_file("./notes.md", &ExtractionConfig::default()).await?;
# Ok(())
# }
```

### Embedding

```rust,no_run
use akuna_core::embedding::{TextEmbedding, TextEmbeddingOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(TextEmbeddingOptions::default()).await?;
    let embedding = model.embed("Hello world")?;
    Ok(())
}
```

### Storage

```rust,no_run
use akuna_core::storage::graph::{
    in_memory_context, GraphDbContext, GraphNode,
};

# fn main() {
let ctx = in_memory_context();
let node = GraphNode {
    id: "rust".to_string(),
    labels: vec!["Concept".to_string(), "Language".to_string()],
    name: "Rust".to_string(),
    description: None,
    metadata: None,
};
ctx.put_node(&node, &[]).expect("node stored");
# }
```
