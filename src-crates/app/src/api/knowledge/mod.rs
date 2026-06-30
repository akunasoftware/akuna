//! Knowledge API routes and handlers.

use std::sync::Arc;

use akuna_core::storage::{
    GraphDbContext, GraphEdge, GraphNode, GraphNodeSearchQuery,
    GraphNodeSearchResult, search_text,
};
use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use serde::Deserialize;

use crate::api::error::{ApiError, ApiErrorBody, ApiResult, ServiceError};

const GRAPH_DB_NAME: &str = "knowledge";

/// Shared router state holding the graph storage backend.
#[derive(Clone)]
pub(crate) struct ApiState {
    graph: Arc<dyn GraphDbContext>,
}

/// Registers knowledge API routes.
pub(crate) fn router() -> Result<Router, ServiceError> {
    Ok(router_with_graph(graph()?))
}

/// Registers knowledge API routes with graph storage.
fn router_with_graph(graph: Box<dyn GraphDbContext>) -> Router {
    let state = ApiState {
        graph: Arc::from(graph),
    };

    Router::new()
        .route("/graph/nodes", post(create_node))
        .route("/graph/nodes/search", get(search_nodes))
        .route(
            "/graph/nodes/{id}",
            get(read_node).put(update_node).delete(delete_node),
        )
        .route(
            "/graph/edges",
            post(create_edge).put(update_edge).delete(delete_edge),
        )
        .with_state(state)
}

/// Query parameters for reading or deleting graph nodes.
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct NodeQuery {
    /// Comma-separated labels scoping node ID.
    labels: String,
}

/// Query parameters for searching graph nodes.
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct NodeSearchQuery {
    /// Search text.
    q: Option<String>,
    /// Optional label to search within.
    labels: Option<String>,
    /// Maximum result count. Defaults to 3. Must be 1 to 50.
    limit: Option<usize>,
}

/// Query parameters identifying a graph edge.
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct EdgeQuery {
    /// Comma-separated source node labels.
    source_labels: String,
    /// Stable source node identifier within its labels.
    source: String,
    /// Relationship type from source to target.
    predicate: String,
    /// Stable target node identifier within its labels.
    target: String,
    /// Comma-separated target node labels.
    target_labels: String,
}

/// Creates a graph node.
#[utoipa::path(
    post,
    path = "/graph/nodes",
    request_body = GraphNode,
    responses(
        (status = 201, description = "Created node", body = GraphNode),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn create_node(
    State(state): State<ApiState>,
    Json(node): Json<GraphNode>,
) -> Result<(StatusCode, Json<GraphNode>), ApiError> {
    validate_node(&node)?;

    write_node(&*state.graph, node)
        .await
        .map(|node| (StatusCode::CREATED, Json(node)))
        .map_err(Into::into)
}

/// Searches graph nodes by text.
#[utoipa::path(
    get,
    path = "/graph/nodes/search",
    params(NodeSearchQuery),
    responses(
        (status = 200, description = "Node search results", body = [GraphNodeSearchResult]),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn search_nodes(
    State(state): State<ApiState>,
    Query(query): Query<NodeSearchQuery>,
) -> ApiResult<Vec<GraphNodeSearchResult>> {
    search_graph_nodes(&*state.graph, query)
        .await
        .map(Json)
        .map_err(Into::into)
}

/// Reads a graph node by ID and labels.
#[utoipa::path(
    get,
    path = "/graph/nodes/{id}",
    params(("id" = String, Path, description = "Stable node ID"), NodeQuery),
    responses(
        (status = 200, description = "Node", body = GraphNode),
        (status = 404, description = "Node not found", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn read_node(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Query(query): Query<NodeQuery>,
) -> ApiResult<GraphNode> {
    read_graph_node(&*state.graph, id, query)
        .map(Json)
        .map_err(Into::into)
}

/// Updates a graph node by ID.
#[utoipa::path(
    put,
    path = "/graph/nodes/{id}",
    params(("id" = String, Path, description = "Stable node ID")),
    request_body = GraphNode,
    responses(
        (status = 200, description = "Updated node", body = GraphNode),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn update_node(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(node): Json<GraphNode>,
) -> ApiResult<GraphNode> {
    validate_node(&node)?;

    if node.id != id {
        return Err(
            ServiceError::bad_request("path id must match node id").into()
        );
    }

    write_node(&*state.graph, node)
        .await
        .map(Json)
        .map_err(Into::into)
}

/// Deletes a graph node by ID and labels.
#[utoipa::path(
    delete,
    path = "/graph/nodes/{id}",
    params(("id" = String, Path, description = "Stable node ID"), NodeQuery),
    responses(
        (status = 204, description = "Deleted node"),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn delete_node(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Query(query): Query<NodeQuery>,
) -> Result<StatusCode, ApiError> {
    delete_graph_node(&*state.graph, id, query)
        .map(|()| StatusCode::NO_CONTENT)
        .map_err(Into::into)
}

/// Creates a graph edge.
#[utoipa::path(
    post,
    path = "/graph/edges",
    request_body = GraphEdge,
    responses(
        (status = 201, description = "Created edge", body = GraphEdge),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn create_edge(
    State(state): State<ApiState>,
    Json(edge): Json<GraphEdge>,
) -> Result<(StatusCode, Json<GraphEdge>), ApiError> {
    validate_edge(&edge)?;

    write_edge(&*state.graph, edge)
        .map(|edge| (StatusCode::CREATED, Json(edge)))
        .map_err(Into::into)
}

/// Updates a graph edge.
#[utoipa::path(
    put,
    path = "/graph/edges",
    request_body = GraphEdge,
    responses(
        (status = 200, description = "Updated edge", body = GraphEdge),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn update_edge(
    State(state): State<ApiState>,
    Json(edge): Json<GraphEdge>,
) -> ApiResult<GraphEdge> {
    validate_edge(&edge)?;

    write_edge(&*state.graph, edge)
        .map(Json)
        .map_err(Into::into)
}

/// Deletes a graph edge by identity.
#[utoipa::path(
    delete,
    path = "/graph/edges",
    params(EdgeQuery),
    responses(
        (status = 204, description = "Deleted edge"),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn delete_edge(
    State(state): State<ApiState>,
    Query(query): Query<EdgeQuery>,
) -> Result<StatusCode, ApiError> {
    state
        .graph
        .delete_edge(&query.into_edge()?)
        .map_err(ServiceError::from)?;

    Ok(StatusCode::NO_CONTENT)
}

/// Stores a graph node.
async fn write_node(
    graph: &dyn GraphDbContext,
    node: GraphNode,
) -> Result<GraphNode, ServiceError> {
    let embedding = embed_search_text(&search_text(&node)).await?;
    graph.put_node(&node, &embedding)?;
    Ok(node)
}

/// Searches graph nodes.
async fn search_graph_nodes(
    graph: &dyn GraphDbContext,
    query: NodeSearchQuery,
) -> Result<Vec<GraphNodeSearchResult>, ServiceError> {
    let Some(query_text) = query.q.as_deref() else {
        return Err(ServiceError::bad_request("q is required"));
    };
    let query_text = query_text.trim();
    if query_text.is_empty() {
        return Err(ServiceError::bad_request("q must not be empty"));
    }

    let limit = query.limit.unwrap_or(3);
    if limit == 0 || limit > 50 {
        return Err(ServiceError::bad_request(
            "limit must be between 1 and 50",
        ));
    }

    let label = match query.labels.as_deref() {
        Some(labels) => {
            let labels = parse_labels(labels)?;
            if labels.len() > 1 {
                return Err(ServiceError::bad_request(
                    "search supports at most one label filter",
                ));
            }

            labels.into_iter().next()
        }
        None => None,
    };
    let embedding = embed_search_text(query_text).await?;
    graph
        .search_nodes(
            &GraphNodeSearchQuery {
                label,
                query: query_text.to_string(),
                limit,
            },
            &embedding,
        )
        .map_err(Into::into)
}

/// Reads a graph node by ID and labels.
fn read_graph_node(
    graph: &dyn GraphDbContext,
    id: String,
    query: NodeQuery,
) -> Result<GraphNode, ServiceError> {
    let labels = parse_labels(&query.labels)?;
    let labels = labels.iter().map(String::as_str).collect::<Vec<_>>();
    let Some(node) = graph.get_node(&labels, &id)? else {
        return Err(ServiceError::not_found("knowledge node not found"));
    };

    Ok(node)
}

/// Deletes a graph node by ID and labels.
fn delete_graph_node(
    graph: &dyn GraphDbContext,
    id: String,
    query: NodeQuery,
) -> Result<(), ServiceError> {
    let labels = parse_labels(&query.labels)?;
    let labels = labels.iter().map(String::as_str).collect::<Vec<_>>();
    graph.delete_node(&labels, &id)?;
    Ok(())
}

/// Stores a graph edge.
fn write_edge(
    graph: &dyn GraphDbContext,
    edge: GraphEdge,
) -> Result<GraphEdge, ServiceError> {
    graph.put_edge(&edge)?;
    Ok(edge)
}

/// Opens graph database context for one API call.
fn graph() -> Result<Box<dyn GraphDbContext>, ServiceError> {
    Ok(akuna_core::storage::open_context(GRAPH_DB_NAME)?)
}

impl EdgeQuery {
    /// Converts query identity into graph edge key.
    fn into_edge(self) -> Result<GraphEdge, ServiceError> {
        let source_labels = parse_labels(&self.source_labels)?;
        let target_labels = parse_labels(&self.target_labels)?;
        validate_graph_identifier(&self.predicate, "predicate")?;

        Ok(GraphEdge {
            source_labels,
            source: self.source,
            predicate: self.predicate,
            target: self.target,
            target_labels,
        })
    }
}

/// Validates graph node fields used in graph query syntax.
fn validate_node(node: &GraphNode) -> Result<(), ServiceError> {
    validate_labels(&node.labels)?;
    let Some(serde_json::Value::Object(metadata)) = node.metadata.as_ref()
    else {
        return Ok(());
    };

    if metadata.keys().any(|key| key.starts_with('_')) {
        return Err(ServiceError::bad_request(
            "metadata keys must not start with underscore",
        ));
    }

    Ok(())
}

/// Validates graph edge fields used in graph query syntax.
fn validate_edge(edge: &GraphEdge) -> Result<(), ServiceError> {
    validate_labels(&edge.source_labels)?;
    validate_labels(&edge.target_labels)?;
    validate_graph_identifier(&edge.predicate, "predicate")
}

/// Parses comma-separated graph labels from query parameters.
fn parse_labels(labels: &str) -> Result<Vec<String>, ServiceError> {
    let labels = labels
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    validate_labels(&labels)?;
    Ok(labels)
}

/// Validates graph labels used in graph query syntax.
fn validate_labels(labels: &[String]) -> Result<(), ServiceError> {
    if labels.is_empty() {
        return Err(ServiceError::bad_request("labels must not be empty"));
    }

    labels
        .iter()
        .try_for_each(|label| validate_graph_identifier(label, "label"))
}

/// Validates identifier shape accepted by graph query syntax.
fn validate_graph_identifier(
    value: &str,
    field: &str,
) -> Result<(), ServiceError> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(ServiceError::bad_request(format!(
            "{field} must not be empty"
        )));
    };

    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(ServiceError::bad_request(format!(
            "{field} must start with a letter or underscore"
        )));
    }

    if chars.all(|char| char.is_ascii_alphanumeric() || char == '_') {
        return Ok(());
    }

    Err(ServiceError::bad_request(format!(
        "{field} must contain only letters, numbers, or underscores"
    )))
}

/// Embeds search text for graph node indexing and querying.
#[cfg(not(test))]
async fn embed_search_text(text: &str) -> Result<Vec<f32>, ServiceError> {
    use tokio::sync::OnceCell;
    static MODEL: OnceCell<akuna_core::embedding::TextEmbedding> =
        OnceCell::const_new();
    let model = MODEL
        .get_or_try_init(|| async {
            akuna_core::embedding::TextEmbedding::new(Default::default()).await
        })
        .await
        .map_err(|source| ServiceError::Internal {
            message: source.to_string(),
        })?;

    let text = text.to_string();
    tokio::task::spawn_blocking(move || model.embed(&text))
        .await
        .map_err(|source| ServiceError::Internal {
            message: source.to_string(),
        })?
        .map_err(|source| ServiceError::Internal {
            message: source.to_string(),
        })
}

/// Embeds search text deterministically in tests.
#[cfg(test)]
async fn embed_search_text(text: &str) -> Result<Vec<f32>, ServiceError> {
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

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Method, Request, StatusCode},
    };
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;

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
        assert!(
            results.body.as_array().expect("results should array").len() <= 3
        );
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
            request(app.clone(), Method::GET, "/graph/nodes/search", None)
                .await;
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

    /// Rejects metadata keys reserved for graph internals.
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
        let response =
            app.oneshot(request).await.expect("request should complete");
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
}
