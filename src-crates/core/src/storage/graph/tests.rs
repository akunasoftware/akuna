use super::{GraphEdge, GraphError, GraphNode, in_memory_context};

/// Lists incoming and outgoing neighbors in a stable order.
#[test]
fn neighbors_both_directions() {
    let ctx = in_memory_context();
    let label = "Record";
    let center = node("center", label);
    let outgoing = node("outgoing", label);
    let incoming = node("incoming", label);
    ctx.put_node(&center).expect("center should store");
    ctx.put_node(&outgoing).expect("outgoing should store");
    ctx.put_node(&incoming).expect("incoming should store");

    ctx.put_edge(&edge(label, "incoming", "center"))
        .expect("incoming edge should store");
    ctx.put_edge(&edge(label, "center", "outgoing"))
        .expect("outgoing edge should store");

    let neighbors = ctx
        .neighbors(&[label], "center")
        .expect("neighbors should read");

    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0].0.source, "center");
    assert_eq!(neighbors[0].1.id, "outgoing");
    assert_eq!(neighbors[1].0.source, "incoming");
    assert_eq!(neighbors[1].1.id, "incoming");
}

/// Lists no neighbors for isolated nodes.
#[test]
fn neighbors_no_edges() {
    let ctx = in_memory_context();
    let label = "Record";
    ctx.put_node(&node("solo", label))
        .expect("node should store");

    let neighbors = ctx
        .neighbors(&[label], "solo")
        .expect("neighbors should read");

    assert!(neighbors.is_empty());
}

/// Lists no neighbors for missing nodes.
#[test]
fn neighbors_missing_node() {
    let ctx = in_memory_context();

    let neighbors = ctx
        .neighbors(&["Record"], "missing")
        .expect("neighbors should read");

    assert!(neighbors.is_empty());
}

/// Errors when storing an edge without both endpoints.
#[test]
fn put_edge_missing_endpoint_errors() {
    let ctx = in_memory_context();
    let label = "Record";
    ctx.put_node(&node("source", label))
        .expect("source should store");

    let error = ctx
        .put_edge(&edge(label, "source", "missing"))
        .expect_err("missing target should fail");

    assert!(matches!(error, GraphError::NotFound { .. }));
}

/// Removes metadata omitted by later upserts.
#[test]
fn put_node_replaces_metadata() {
    let ctx = in_memory_context();
    let label = "Record";
    let mut first = node("record", label);
    first.metadata = Some(serde_json::json!({ "color": "red" }));
    ctx.put_node(&first).expect("first node should store");

    ctx.put_node(&node("record", label))
        .expect("second node should store");
    let stored = ctx
        .get_node(&[label], "record")
        .expect("node should read")
        .expect("node should exist");

    assert_eq!(stored.metadata, None);
}

/// Retains null metadata values across storage round trips.
#[test]
fn put_node_preserves_null_metadata() {
    let ctx = in_memory_context();
    let label = "Record";
    let mut node = node("record", label);
    node.metadata = Some(serde_json::json!({ "nullable": null }));

    ctx.put_node(&node).expect("node should store");
    let stored = ctx.get_node(&[label], "record").expect("node should read");

    assert_eq!(stored.map(|node| node.metadata), Some(node.metadata));
}

/// Requires exact labels for node identity.
#[test]
fn get_node_requires_exact_labels() {
    let ctx = in_memory_context();
    let extra = GraphNode {
        labels: vec![
            "Record".to_string(),
            "docs".to_string(),
            "Extra".to_string(),
        ],
        ..node("record", "Record")
    };
    ctx.put_node(&extra).expect("node should store");

    assert!(
        ctx.get_node(&["Record", "docs"], "record")
            .expect("node should read")
            .is_none(),
    );
    assert!(
        ctx.get_node(&["Record", "docs", "Extra"], "record")
            .expect("node should read")
            .is_some(),
    );
}

/// Builds a test graph node.
fn node(id: &str, label: &str) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        labels: vec![label.to_string()],
        name: id.to_string(),
        description: None,
        metadata: None,
    }
}

/// Builds a test graph edge.
fn edge(label: &str, source: &str, target: &str) -> GraphEdge {
    GraphEdge {
        source_labels: vec![label.to_string()],
        source: source.to_string(),
        predicate: "RELATED_TO".to_string(),
        target: target.to_string(),
        target_labels: vec![label.to_string()],
    }
}
