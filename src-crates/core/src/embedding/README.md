# Embedding Namespace

`akuna_core::embedding` loads a shared text embedding model and embeds text.
Feature-gated behind `embedding`.

Use `model` to load the default model once and reuse it across calls.
Use `embed_batch` when embedding multiple texts.

## How To Use It

Use `embedding::model()` for shared application-wide embedding.
It initializes the default model once, then reuses it through a `OnceCell`.
Use `TextEmbedding::new` when you need explicit model options or cache location.
Use `embed_batch` for multiple inputs and pass a batch size when you need to control throughput or memory use.

| API                    | Purpose                                            |
| ---------------------- | -------------------------------------------------- |
| `model()`              | Shared default embedding model.                    |
| `TextEmbedding::new`   | Custom model/options instance.                     |
| `embed`                | Embed one input string.                            |
| `embed_batch`          | Embed many input strings.                          |
| `TextEmbeddingOptions` | Choose model variant and optional cache directory. |

## Custom Model

```rust
use std::path::PathBuf;

use akuna_core::embedding::{EmbeddingModel, TextEmbedding, TextEmbeddingOptions};

let options = TextEmbeddingOptions {
    model: EmbeddingModel::MiniLmL12,
    cache_dir: Some(PathBuf::from("./models")),
};

let model = TextEmbedding::new(options).await?;
```

## Batch Embedding

```rust
use akuna_core::embedding;

let model = embedding::model().await?;

let embeddings = model.embed_batch(
    &["Hello world", "Rust embeddings"],
    Some(2),
)?;
```
