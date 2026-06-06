# Text Reranking

Cross-encoder text reranking models built on [Burn](https://github.com/tracel-ai/burn).

## Usage

```rust
use akuna_core::reranking::TextReranker;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::try_new().await?;

    // `score` is blocking. In a real service, wrap heavy inference in
    // `tokio::task::spawn_blocking` to avoid stalling async workers.
    let score = model.score("Rust ML", "Burn is a Rust deep learning framework")?;
    println!("Relevance score: {score}");

    Ok(())
}
```

## Score Many Pairs

```rust
use akuna_core::reranking::TextReranker;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::try_new().await?;

    // `score_batch` is blocking; see the Usage section above.
    let scores = model.score_batch(
        &[
            ("Rust ML", "Burn is a Rust deep learning framework"),
            ("Rust ML", "Bananas are yellow"),
        ],
        None,
    )?;

    println!("{scores:?}");

    Ok(())
}
```

## Rerank Documents

```rust
use akuna_core::reranking::{RerankOptions, TextReranker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::try_new().await?;

    // `rerank_with_options` is blocking; see the Usage section above.
    let results = model.rerank_with_options(
        "Rust ML",
        &[
            "Burn is a Rust deep learning framework",
            "Bananas are yellow",
            "Rust has strong typing",
        ],
        RerankOptions {
            top_k: Some(2),
            normalize: true,
            batch_size: None,
        },
    )?;

    println!("Best document index: {}", results[0].index);
    println!("Best document: {}", results[0].document);

    Ok(())
}
```

`normalize: true` applies sigmoid to raw logits, mapping scores to `[0, 1]`.

## Models

The built-in reranker is `BAAI/bge-reranker-base`.
