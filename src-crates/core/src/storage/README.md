# Graph Storage

Graph storage and retrieval built on `grafeo`.

## Usage

```rust
use akuna_core::storage::{in_memory_context, GraphDbContext, GraphNode};

let ctx = in_memory_context();

let node = GraphNode {
    id: "42".to_string(),
    labels: vec!["Concept".to_string()],
    name: "Example".to_string(),
    description: Some("A sample node".to_string()),
    metadata: None,
};

ctx.put_node(&node, &[0.1, 0.2, 0.3]).expect("node stored");
```

For persistent storage, use `open_context` with a filesystem path.

## Overview

The public surface exposes:

- The backend-neutral `GraphDbContext` trait.
- Domain types for nodes, edges, and search queries.
- Constructors for in-memory and persistent contexts.

Backend implementations are private and reachable only through `open_context`
or `in_memory_context`.

## Features

- `api`: derives `utoipa::ToSchema` on public response types.
