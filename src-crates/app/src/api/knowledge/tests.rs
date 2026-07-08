use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use akuna_core::index::{
    Index, IndexOptions, Metadata, MetadataValue, Record, RecordRelationship,
};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode},
};
use serde_json::Value;
use tokio::sync::OnceCell;
use tower::ServiceExt;

use crate::api::knowledge::router_with_index;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static INDEXES: OnceCell<TestIndexes> = OnceCell::const_new();

struct TestIndexes {
    slim: Arc<Index>,
    full: Arc<Index>,
}

/// Runs record CRUD through HTTP handlers.
#[tokio::test]
async fn records_crud() -> TestResult {
    let app = slim_router().await;
    let collection = test_name("crud");
    let id = test_name("record");
    let original = record(
        &collection,
        &id,
        "Original",
        "old body",
        Metadata::new(),
        Vec::new(),
    );

    let created = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(records_body(vec![original])?),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK);
    assert_eq!(created.body[0]["id"], id);
    assert_eq!(created.body[0]["content"], "old body");

    let read = request(
        app.clone(),
        Method::GET,
        &format!("/records/{collection}/{id}"),
        None,
    )
    .await;
    assert_eq!(read.status, StatusCode::OK);
    assert_eq!(read.body["title"], "Original");

    let updated = record(
        &collection,
        &id,
        "Updated",
        "new body",
        Metadata::new(),
        Vec::new(),
    );
    let upserted = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(records_body(vec![updated])?),
    )
    .await;
    assert_eq!(upserted.status, StatusCode::OK);
    assert_eq!(upserted.body[0]["title"], "Updated");

    let replaced = request(
        app.clone(),
        Method::GET,
        &format!("/records/{collection}/{id}"),
        None,
    )
    .await;
    assert_eq!(replaced.status, StatusCode::OK);
    assert_eq!(replaced.body["content"], "new body");

    let deleted = request(
        app.clone(),
        Method::DELETE,
        &format!("/records/{collection}/{id}"),
        None,
    )
    .await;
    assert_eq!(deleted.status, StatusCode::NO_CONTENT);

    let missing = request(
        app.clone(),
        Method::GET,
        &format!("/records/{collection}/{id}"),
        None,
    )
    .await;
    assert_eq!(missing.status, StatusCode::NOT_FOUND);
    assert_eq!(missing.body["error"], "not_found");

    let delete_missing = request(
        app,
        Method::DELETE,
        &format!("/records/{collection}/missing"),
        None,
    )
    .await;
    assert_eq!(delete_missing.status, StatusCode::NO_CONTENT);

    Ok(())
}

/// Keeps records scoped by collection.
#[tokio::test]
async fn records_collection_scope() -> TestResult {
    let app = slim_router().await;
    let left_collection = test_name("left");
    let right_collection = test_name("right");
    let id = test_name("same");

    let created = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(records_body(vec![
            record(
                &left_collection,
                &id,
                "Left",
                "left body",
                Metadata::new(),
                Vec::new(),
            ),
            record(
                &right_collection,
                &id,
                "Right",
                "right body",
                Metadata::new(),
                Vec::new(),
            ),
        ])?),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK);

    let left = request(
        app.clone(),
        Method::GET,
        &format!("/records/{left_collection}/{id}"),
        None,
    )
    .await;
    let right = request(
        app,
        Method::GET,
        &format!("/records/{right_collection}/{id}"),
        None,
    )
    .await;

    assert_eq!(left.status, StatusCode::OK);
    assert_eq!(right.status, StatusCode::OK);
    assert_eq!(left.body["title"], "Left");
    assert_eq!(right.body["title"], "Right");

    Ok(())
}

/// Searches records with collection and metadata filters.
#[tokio::test]
async fn records_search_filter() -> TestResult {
    let app = slim_router().await;
    let docs = test_name("docs");
    let notes = test_name("notes");
    let token = format!("shared search {}", unique_suffix());

    let created = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(records_body(vec![
            record(
                &docs,
                "red_doc",
                "Red Doc",
                &token,
                metadata_text("color", "red"),
                Vec::new(),
            ),
            record(
                &docs,
                "blue_doc",
                "Blue Doc",
                &token,
                metadata_text("color", "blue"),
                Vec::new(),
            ),
            record(
                &notes,
                "red_note",
                "Red Note",
                &token,
                metadata_text("color", "red"),
                Vec::new(),
            ),
        ])?),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK);

    let searched = request(
        app,
        Method::GET,
        &format!(
            "/records/search?q=shared%20search&collections={docs}&limit=10&filter={}",
            red_filter()
        ),
        None,
    )
    .await;

    assert_eq!(searched.status, StatusCode::OK);
    assert_eq!(
        searched
            .body
            .as_array()
            .expect("search body should be an array")
            .iter()
            .map(|result| {
                result["record_id"]
                    .as_str()
                    .expect("result should have record_id")
            })
            .collect::<Vec<_>>(),
        vec!["red_doc"]
    );

    Ok(())
}

/// Rejects invalid search query parameters.
#[tokio::test]
async fn records_search_bad_requests() {
    let app = slim_router().await;
    for uri in [
        "/records/search?limit=1",
        "/records/search?q=test&limit=abc",
        "/records/search?q=%20",
        "/records/search?q=test&limit=0",
        "/records/search?q=test&filter=%7Bbad",
    ] {
        let response = request(app.clone(), Method::GET, uri, None).await;
        assert_eq!(response.status, StatusCode::BAD_REQUEST);
        assert_eq!(response.body["error"], "bad_request");
    }
}

/// Rejects invalid record write requests with the API error body.
#[tokio::test]
async fn records_write_bad_requests() -> TestResult {
    let app = full_router().await;
    let malformed = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(Value::String("not records".to_string())),
    )
    .await;
    assert_eq!(malformed.status, StatusCode::BAD_REQUEST);
    assert_eq!(malformed.body["error"], "bad_request");

    let collection = test_name("missing_target");
    let id = test_name("record");
    let missing_target = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(records_body(vec![record(
            &collection,
            &id,
            "Missing Target",
            "body",
            Metadata::new(),
            vec![RecordRelationship {
                predicate: "related-to".to_string(),
                record_id: "missing".to_string(),
                collection: collection.clone(),
            }],
        )])?),
    )
    .await;
    assert_eq!(missing_target.status, StatusCode::BAD_REQUEST);
    assert_eq!(missing_target.body["error"], "bad_request");

    let read = request(
        app,
        Method::GET,
        &format!("/records/{collection}/{id}"),
        None,
    )
    .await;
    assert_eq!(read.status, StatusCode::NOT_FOUND);

    Ok(())
}

/// Exercises default search pipeline through HTTP.
#[tokio::test]
async fn records_search_full_pipeline() -> TestResult {
    let app = full_router().await;
    let collection = test_name("full");
    let source = test_name("source");
    let target = test_name("target");
    let filler = (0..90)
        .map(|index| format!("filler{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let passage = "volcano needle passage";
    let content = format!("{filler} {passage} {filler}");

    let created = request(
        app.clone(),
        Method::POST,
        "/records",
        Some(records_body(vec![
            record(
                &collection,
                &source,
                "Source",
                &content,
                Metadata::new(),
                vec![RecordRelationship {
                    predicate: "cites".to_string(),
                    record_id: target.clone(),
                    collection: collection.clone(),
                }],
            ),
            record(
                &collection,
                &target,
                "Target",
                "expanded target leading body",
                Metadata::new(),
                Vec::new(),
            ),
        ])?),
    )
    .await;
    assert_eq!(created.status, StatusCode::OK);

    let searched = request(
        app,
        Method::GET,
        &format!(
            "/records/search?q=volcano%20needle&collections={collection}&limit=2"
        ),
        None,
    )
    .await;
    assert_eq!(searched.status, StatusCode::OK);

    let source_result = search_result(&searched.body, &source);
    let target_result = search_result(&searched.body, &target);
    assert!(
        source_result["preview"]
            .as_str()
            .is_some_and(|preview| { preview.contains(passage) })
    );
    assert!(
        target_result["preview"]
            .as_str()
            .is_some_and(|preview| { !preview.is_empty() })
    );

    Ok(())
}

struct TestResponse {
    status: StatusCode,
    body: Value,
}

/// Builds API router backed by shared slim test index.
async fn slim_router() -> Router {
    router_with_index(slim_index().await)
}

/// Returns shared test indexes.
async fn test_indexes() -> &'static TestIndexes {
    INDEXES
        .get_or_init(|| async {
            let slim = Arc::new(
                Index::new(IndexOptions {
                    reranking_model: None,
                    fulltext: false,
                    ..Default::default()
                })
                .await
                .expect("slim index should build"),
            );
            let full = Arc::new(
                Index::new(Default::default())
                    .await
                    .expect("full index should build"),
            );

            TestIndexes { slim, full }
        })
        .await
}

/// Returns the shared slim test index.
async fn slim_index() -> Arc<Index> {
    test_indexes().await.slim.clone()
}

/// Builds API router backed by a default index.
async fn full_router() -> Router {
    router_with_index(test_indexes().await.full.clone())
}

/// Builds a test record.
fn record(
    collection: &str,
    id: &str,
    title: &str,
    content: &str,
    metadata: Metadata,
    relationships: Vec<RecordRelationship>,
) -> Record {
    Record {
        id: id.to_string(),
        collection: collection.to_string(),
        title: title.to_string(),
        content: content.to_string(),
        metadata,
        relationships,
    }
}

/// Builds text metadata.
fn metadata_text(key: &str, value: &str) -> Metadata {
    [(key.to_string(), MetadataValue::Text(value.to_string()))]
        .into_iter()
        .collect()
}

/// Serializes records as a JSON request body.
fn records_body(records: Vec<Record>) -> TestResult<Value> {
    Ok(serde_json::to_value(records)?)
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
        match serde_json::from_slice(&body) {
            Ok(body) => body,
            Err(_error) => Value::Null,
        }
    };

    TestResponse { status, body }
}

/// Finds one search result by record id.
fn search_result<'a>(body: &'a Value, record_id: &str) -> &'a Value {
    body.as_array()
        .expect("search body should be an array")
        .iter()
        .find(|result| result["record_id"] == record_id)
        .expect("search result should exist")
}

/// URL-encoded metadata filter for color red.
fn red_filter() -> &'static str {
    "%7B%22equals%22%3A%7B%22key%22%3A%22color%22%2C%22value%22%3A%7B%22text%22%3A%22red%22%7D%7D%7D"
}

/// Builds a unique test name.
fn test_name(prefix: &str) -> String {
    format!("{prefix}_{}", unique_suffix())
}

/// Returns a unique suffix.
fn unique_suffix() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
