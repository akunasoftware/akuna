# Text Embeddings

Text embedding models (MiniLM, BGE, MPNet, BGE-M3) built on [Burn](https://github.com/tracel-ai/burn).

## Usage

```rust
use akuna_core::embedding::{TextEmbedding, TextEmbeddingOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(TextEmbeddingOptions::default()).await?;

    // `embed` is blocking. In a real service, wrap heavy inference in
    // `tokio::task::spawn_blocking` to avoid stalling async workers.
    let single = model.embed("Hello world")?;
    let batch = model.embed_batch(&["Hello world", "Rust embeddings"], None)?;

    println!("single: {}, batch: {}", single.len(), batch.len());

    Ok(())
}
```

## Search

Embed stored content with `embed` or `embed_batch`.
Embed user queries with `embed_query` or `embed_query_batch`.
The query methods follow `sentence-transformers` defaults and add no hidden prompts.

If a model card recommends a prompt, pass it explicitly via
`embed_query_with_prompt` or `embed_query_batch_with_prompt`.

```rust,no_run
use akuna_core::embedding::TextEmbedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(Default::default()).await?;
    let prompt = "Represent this sentence for searching relevant passages: ";

    // `embed*` calls are blocking; see the Usage section above.
    let document = model.embed("Burn is a deep learning framework for Rust")?;
    let query = model.embed_query_with_prompt("Rust machine learning", Some(prompt))?;

    assert_eq!(document.len(), query.len());

    Ok(())
}
```

## Models

`EmbeddingModel::MiniLmL12` is the default.

| Variant              | Checkpoint                                  | Dimensions |
| -------------------- | ------------------------------------------- | ---------- |
| `MiniLmL6`           | `sentence-transformers/all-MiniLM-L6-v2`    | 384        |
| `MiniLmL12`          | `sentence-transformers/all-MiniLM-L12-v2`   | 384        |
| `BgeSmallEnV15`      | `BAAI/bge-small-en-v1.5`                    | 384        |
| `BgeBaseEnV15`       | `BAAI/bge-base-en-v1.5`                     | 768        |
| `BgeLargeEnV15`      | `BAAI/bge-large-en-v1.5`                    | 1024       |
| `AllMpnetBaseV2`     | `sentence-transformers/all-mpnet-base-v2`   | 768        |
| `BgeM3`              | `BAAI/bge-m3`                               | 1024       |

`BgeM3` exposes dense embeddings only.
Sparse and multi-vector outputs are separate retrieval concerns
and are not part of the `Vec<f32>` API.

```rust,no_run
use akuna_core::embedding::{EmbeddingModel, TextEmbedding, TextEmbeddingOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextEmbedding::new(TextEmbeddingOptions {
        model: EmbeddingModel::BgeSmallEnV15,
        ..Default::default()
    })
    .await?;

    // `embed` is blocking; see the Usage section above.
    let embedding = model.embed("Hello world")?;
    assert!(!embedding.is_empty());

    Ok(())
}
```
