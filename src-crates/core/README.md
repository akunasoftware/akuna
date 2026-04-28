<h1>
  <img src="../../assets/icon-gradient.svg" alt="" width="36" align="left">
  Akuna Core Library
</h1>

Reusable library APIs for document extraction, text chunking, embeddings, and graph storage.

See the [workspace README](../../README.md) for feature overview.

## Extraction

Use `extract_file` for the high-level extraction flow.
The default config returns metadata only.

#### Minimal Example. See [`akuna_core::extraction`](./src/extraction/) for more details.

```rust
use akuna_core::{ExtractionConfig, extraction::extract_file};

let result = extract_file("./notes.md", &ExtractionConfig::default()).await?;
```

## Chunking

Use `chunk_text` directly when content is already available.

#### Minimal Example. See [`akuna_core::chunking`](./src/chunking/) for more details.

```rust
use akuna_core::chunking::chunk_text;

let chunks = chunk_text(None, "hello\nworld", Some("txt"));
```

## Embedding

Use `model` to load the default model once and reuse it across calls.

#### Minimal Example. See [`akuna_core::embedding`](./src/embedding/) for more details.

```rust
use akuna_core::embedding;

let model = embedding::model().await?;

let embedding = model.embed("Hello world")?;
```

## Graph

Use [`graph`](./src/graph/) when storing domain data as typed graph nodes and edges.

#### Minimal Example. See [`akuna_core::graph`](./src/graph/) for more details.

```rust
use akuna_core::graph::knowledge::Concept;

let concept = Concept {
    id: "rust".to_string(),
    labels: vec!["Concept".to_string(), "Language".to_string()],
    name: "Rust".to_string(),
    description: None,
    metadata: None,
};
```
