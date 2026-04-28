# Graph Namespace

`akuna_core::graph` helps build graph-backed models without storage-specific code.
Feature-gated behind `graph`.

| Module       | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `knowledge`  | Ready-made knowledge graph node and edge types.            |
| `primitives` | Traits and derive macros for custom domain graph types.    |
| `storage`    | Backend adapters that implement the shared graph contract. |

## How To Use It

Pick the highest-level shape that fits your data.
Use `knowledge` for generic knowledge graphs.
Use `primitives` when your domain structs should be graph nodes or edges.
Use `storage` only when wiring a concrete backend.

Most application code should accept `GraphDbContext` so storage remains swappable.
Node ids are scoped by label set, so read and delete calls must pass matching labels.

## Knowledge Types

Use `knowledge` when the generic graph shapes match the data being stored.
`Concept` and `Relationship` are ready for general graph use.
`Assertion` and `Provenance` are public but still work in progress.

```rust
use akuna_core::graph::knowledge::{Concept, Relationship};
use serde_json::json;

let concept = Concept {
    id: "rust".to_string(),
    labels: vec!["Concept".to_string(), "Language".to_string()],
    name: "Rust".to_string(),
    description: Some("Systems programming language".to_string()),
    metadata: Some(json!({ "first_release_year": 2015 })),
};

let relationship = Relationship {
    source_labels: vec!["Concept".to_string(), "Language".to_string()],
    source: "rust".to_string(),
    predicate: "influenced_by".to_string(),
    target: "ml".to_string(),
    target_labels: vec!["Concept".to_string(), "Language".to_string()],
};
```

## Custom Types

Use `primitives` when domain structs need their own graph shape.
Derive `GraphNode` and `GraphEdge`, then mark fields that map to the storage contract.

```rust
use akuna_core::graph::primitives::{GraphEdge, GraphNode};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, GraphNode, PartialEq, Deserialize, Serialize)]
struct Person {
    #[graph(id)]
    id: String,
    #[graph(labels)]
    labels: Vec<String>,
    #[graph(name)]
    name: String,
    #[graph(description)]
    description: Option<String>,
    #[graph(metadata)]
    metadata: Option<serde_json::Value>,
}

#[derive(Clone, Debug, GraphEdge, PartialEq, Eq)]
struct ReportsTo {
    #[graph(source_labels)]
    source_labels: Vec<String>,
    #[graph(source)]
    source: String,
    #[graph(predicate)]
    predicate: String,
    #[graph(target)]
    target: String,
    #[graph(target_labels)]
    target_labels: Vec<String>,
}
```

## Storage

Use `storage` when wiring the concrete backend.
Most domain code should depend on `GraphDbContext`, not a storage adapter directly.
Node ids are scoped by label set, so pass the same labels when reading or deleting nodes.
This example continues from the custom `Person` and `ReportsTo` types above.

```rust
use akuna_core::graph::{
    primitives::GraphDbContext,
    storage::grafeo::GrafeoDbContext,
};

let graph_db = GrafeoDbContext::new_in_memory();

graph_db.put_node(&person)?;
graph_db.put_edge(&reports_to)?;

let person = graph_db.get_node::<Person>(&["Person"], "person-1")?;
```
