use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::api::error::ServiceError;

use super::router_with_graph;

/// Embeds search text deterministically in tests.
pub(super) async fn embed_search_text(
    text: &str,
) -> Result<Vec<f32>, ServiceError> {
    let lowercase = text.to_ascii_lowercase();
    let graph = if lowercase.contains("graph") {
        1.0
    } else {
        0.0
    };
    let database = if lowercase.contains("database") {
        1.0
    } else {
        0.0
    };
    let rust = if lowercase.contains("rust") { 1.0 } else { 0.0 };

    Ok(vec![graph, database, rust])
}

/// Runs node CRUD through HTTP handlers.
#[tokio::test]
async fn node_crud() {
    let label = test_label();
    let id = format!("{}_node", unique_suffix());
    let app = test_router();

    let created = request(
        app.clone(),
        Method::POST,
        "/graph/nodes",
        Some(json!({
            "id": id,
            "labels": [label],
            "name": "Node",
            "description": "created",
            "metadata": {"kind": "source"}
        })),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED);
    assert_eq!(created.body["name"], "Node");

    let read = request(
        app.clone(),
        Method::GET,
        &format!("/graph/nodes/{id}?labels={label}"),
        None,
    )
    .await;
    assert_eq!(read.status, StatusCode::OK);
    assert_eq!(read.body["id"], id);

    let updated = request(
        app.clone(),
        Method::PUT,
        &format!("/graph/nodes/{id}"),
        Some(json!({
            "id": id,
            "labels": [label],
            "name": "Node Updated",
            "description": "updated",
            "metadata": {"kind": "source"}
        })),
    )
    .await;
    assert_eq!(updated.status, StatusCode::OK);
    assert_eq!(updated.body["name"], "Node Updated");

    let deleted = request(
        app,
        Method::DELETE,
        &format!("/graph/nodes/{id}?labels={label}"),
        None,
    )
    .await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);
}

/// Runs edge CRUD through HTTP handlers.
#[tokio::test]
async fn edge_crud() {
    let label = test_label();
    let suffix = unique_suffix();
    let source = format!("{suffix}_source");
    let target = format!("{suffix}_target");
    let app = test_router();

    create_node_for_edge(app.clone(), &source, &label).await;
    create_node_for_edge(app.clone(), &target, &label).await;

    let edge = json!({
        "source_labels": [label],
        "source": source,
        "predicate": "RELATED_TO",
        "target": target,
        "target_labels": [label]
    });

    let created = request(
        app.clone(),
        Method::POST,
        "/graph/edges",
        Some(edge.clone()),
    )
    .await;
    assert_eq!(created.status, StatusCode::CREATED);
    assert_eq!(created.body["predicate"], "RELATED_TO");

    let updated =
        request(app.clone(), Method::PUT, "/graph/edges", Some(edge)).await;
    assert_eq!(updated.status, StatusCode::OK);

    let deleted = request(
        app.clone(),
        Method::DELETE,
        &format!(
            "/graph/edges?source_labels={label}&source={source}&predicate=RELATED_TO&target={target}&target_labels={label}"
        ),
        None,
    )
    .await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);

    cleanup_node(app.clone(), &source, &label).await;
    cleanup_node(app, &target, &label).await;
}

/// Searches nodes with default limit.
#[tokio::test]
async fn node_search() {
    let label = test_label();
    let suffix = unique_suffix();
    let graph_id = format!("{suffix}_graph");
    let rust_id = format!("{suffix}_rust");
    let app = test_router();

    let graph_node = request(
        app.clone(),
        Method::POST,
        "/graph/nodes",
        Some(json!({
            "id": graph_id,
            "labels": [label],
            "name": "Graph Database",
            "description": "native hybrid search",
            "metadata": null
        })),
    )
    .await;
    assert_eq!(graph_node.status, StatusCode::CREATED);

    let rust_node = request(
        app.clone(),
        Method::POST,
        "/graph/nodes",
        Some(json!({
            "id": rust_id,
            "labels": [label],
            "name": "Rust Language",
            "description": "systems programming",
            "metadata": null
        })),
    )
    .await;
    assert_eq!(rust_node.status, StatusCode::CREATED);

    let results = request(
        app.clone(),
        Method::GET,
        "/graph/nodes/search?q=graph",
        None,
    )
    .await;
    assert_eq!(results.status, StatusCode::OK);
    assert!(results.body.as_array().expect("results should array").len() <= 3);
    assert_eq!(results.body[0]["node"]["id"], graph_id);

    let invalid = request(
        app.clone(),
        Method::GET,
        &format!("/graph/nodes/search?q=graph&labels={label}&limit=0"),
        None,
    )
    .await;
    assert_eq!(invalid.status, StatusCode::BAD_REQUEST);
    assert_eq!(invalid.body["error"], "bad_request");

    let missing_query =
        request(app.clone(), Method::GET, "/graph/nodes/search", None).await;
    assert_eq!(missing_query.status, StatusCode::BAD_REQUEST);
    assert_eq!(missing_query.body["error"], "bad_request");

    let malformed_limit = request(
        app.clone(),
        Method::GET,
        &format!("/graph/nodes/search?q=graph&labels={label}&limit=bad"),
        None,
    )
    .await;
    assert_eq!(malformed_limit.status, StatusCode::BAD_REQUEST);

    cleanup_node(app.clone(), &graph_id, &label).await;
    cleanup_node(app, &rust_id, &label).await;
}

/// Rejects labels that cannot safely map to graph query identifiers.
#[tokio::test]
async fn invalid_label_is_bad_request() {
    let app = test_router();
    let response = request(
        app,
        Method::POST,
        "/graph/nodes",
        Some(json!({
            "id": "invalid_label_node",
            "labels": ["bad-label"],
            "name": "Node",
            "description": null,
            "metadata": null
        })),
    )
    .await;

    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.body["error"], "bad_request");
}

// Rejects metadata keys reserved for graph internals.
#[tokio::test]
async fn reserved_metadata_is_bad_request() {
    let app = test_router();
    let response = request(
        app,
        Method::POST,
        "/graph/nodes",
        Some(json!({
            "id": "reserved_metadata_node",
            "labels": [test_label()],
            "name": "Node",
            "description": null,
            "metadata": {"_id": "shadow"}
        })),
    )
    .await;

    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.body["error"], "bad_request");
}

struct TestResponse {
    status: StatusCode,
    body: Value,
}

/// Builds API router for tests.
fn test_router() -> Router {
    router_with_graph(akuna_core::storage::in_memory_context())
}

/// Creates node required for edge tests.
async fn create_node_for_edge(app: Router, id: &str, label: &str) {
    let response = request(
        app,
        Method::POST,
        "/graph/nodes",
        Some(json!({
            "id": id,
            "labels": [label],
            "name": id,
            "description": null,
            "metadata": null
        })),
    )
    .await;
    assert_eq!(response.status, StatusCode::CREATED);
}

/// Deletes node after edge tests.
async fn cleanup_node(app: Router, id: &str, label: &str) {
    let response = request(
        app,
        Method::DELETE,
        &format!("/graph/nodes/{id}?labels={label}"),
        None,
    )
    .await;
    assert_eq!(response.status, StatusCode::NO_CONTENT);
}

/// Sends JSON request to API router.
async fn request(
    app: Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> TestResponse {
    let body =
        body.map_or_else(Body::empty, |body| Body::from(body.to_string()));
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body)
        .expect("request should build");
    let response = app.oneshot(request).await.expect("request should complete");
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body should read");
    let body = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).unwrap_or(Value::Null)
    };

    TestResponse { status, body }
}

/// Builds valid graph label for tests.
fn test_label() -> String {
    format!("ApiTest_{}", unique_suffix())
}

/// Builds unique suffix for persistent graph tests.
fn unique_suffix() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time should be after epoch")
        .as_nanos()
        .to_string()
}
