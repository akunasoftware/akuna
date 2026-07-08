//! Knowledge API routes and handlers.

use std::{path::PathBuf, sync::Arc};

use akuna_core::index::{
    Index, IndexOptions, IndexSearchQuery, IndexSearchResult, MetadataFilter,
    Record,
};
use axum::{
    Json, Router,
    extract::{
        Path, Query, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::StatusCode,
    routing::{get, post},
};
use directories::ProjectDirs;
use serde::Deserialize;

use crate::api::error::{ApiError, ApiErrorBody, ApiResult, ServiceError};

const INDEX_NAME: &str = "knowledge";
const MAX_SEARCH_LIMIT: usize = 100;

/// Shared router state holding the record index.
#[derive(Clone)]
pub(crate) struct ApiState {
    index: Arc<Index>,
}

/// Registers knowledge API routes.
pub(crate) async fn router() -> Result<Router, ServiceError> {
    let index = Index::new(IndexOptions {
        name: INDEX_NAME.to_string(),
        path: Some(data_root()?),
        ..Default::default()
    })
    .await?;

    Ok(router_with_index(Arc::new(index)))
}

/// Registers knowledge API routes with an existing index.
pub(crate) fn router_with_index(index: Arc<Index>) -> Router {
    Router::new()
        .route("/records", post(upsert_records))
        .route("/records/search", get(search_records))
        .route(
            "/records/{collection}/{id}",
            get(read_record).delete(delete_record),
        )
        .with_state(ApiState { index })
}

/// Query parameters for searching records.
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct SearchQuery {
    /// Query text.
    q: String,
    /// Comma-separated collections to search.
    collections: Option<String>,
    /// Maximum result count.
    limit: Option<usize>,
    /// URL-encoded JSON MetadataFilter.
    filter: Option<String>,
}

/// Adds or replaces records.
#[utoipa::path(
    post,
    path = "/records",
    request_body(content = [Record], description = "Records to upsert"),
    responses(
        (status = 200, description = "Upserted records", body = [Record]),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Index operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn upsert_records(
    State(state): State<ApiState>,
    records: Result<Json<Vec<Record>>, JsonRejection>,
) -> ApiResult<Vec<Record>> {
    let Json(records) = records.map_err(bad_request_rejection)?;
    state
        .index
        .add(records.clone())
        .await
        .map_err(ServiceError::from)?;
    Ok(Json(records))
}

/// Returns one record.
#[utoipa::path(
    get,
    path = "/records/{collection}/{id}",
    params(
        ("collection" = String, Path, description = "Record collection"),
        ("id" = String, Path, description = "Stable record ID"),
    ),
    responses(
        (status = 200, description = "Record", body = Record),
        (status = 404, description = "Record not found", body = ApiErrorBody),
        (status = 500, description = "Index operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn read_record(
    State(state): State<ApiState>,
    Path((collection, id)): Path<(String, String)>,
) -> ApiResult<Record> {
    let record = state
        .index
        .get(&collection, &id)
        .await
        .map_err(ServiceError::from)?
        .ok_or_else(|| ServiceError::not_found("knowledge record not found"))?;

    Ok(Json(record))
}

/// Deletes one record.
#[utoipa::path(
    delete,
    path = "/records/{collection}/{id}",
    params(
        ("collection" = String, Path, description = "Record collection"),
        ("id" = String, Path, description = "Stable record ID"),
    ),
    responses(
        (status = 204, description = "Deleted record"),
        (status = 500, description = "Index operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn delete_record(
    State(state): State<ApiState>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<StatusCode, ApiError> {
    state
        .index
        .remove(&collection, &id)
        .await
        .map_err(ServiceError::from)?;

    Ok(StatusCode::NO_CONTENT)
}

/// Searches records.
#[utoipa::path(
    get,
    path = "/records/search",
    params(SearchQuery),
    responses(
        (status = 200, description = "Search results", body = [IndexSearchResult]),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Index operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn search_records(
    State(state): State<ApiState>,
    query: Result<Query<SearchQuery>, QueryRejection>,
) -> ApiResult<Vec<IndexSearchResult>> {
    let Query(query) = query.map_err(bad_request_rejection)?;
    let results = state
        .index
        .search(query.into_index_query()?)
        .await
        .map_err(ServiceError::from)?;

    Ok(Json(results))
}

/// Converts extractor failures into the API error contract.
fn bad_request_rejection(error: impl std::fmt::Display) -> ServiceError {
    ServiceError::bad_request(error.to_string())
}

impl SearchQuery {
    /// Converts HTTP query parameters into an index search query.
    fn into_index_query(self) -> Result<IndexSearchQuery, ServiceError> {
        let text = self.q.trim().to_string();
        if text.is_empty() {
            return Err(ServiceError::bad_request("q must not be empty"));
        }

        Ok(IndexSearchQuery {
            text,
            collections: parse_collections(self.collections),
            filter: self
                .filter
                .map(|filter| serde_json::from_str::<MetadataFilter>(&filter))
                .transpose()?,
            limit: search_limit(self.limit)?,
        })
    }
}

/// Resolves the platform data root for the app.
fn data_root() -> Result<PathBuf, ServiceError> {
    ProjectDirs::from("", "", crate::APP_NAME)
        .map(|dirs| dirs.data_dir().to_path_buf())
        .ok_or_else(|| {
            ServiceError::internal(
                "failed to resolve application data directory",
            )
        })
}

/// Parses comma-separated collection names.
fn parse_collections(collections: Option<String>) -> Vec<String> {
    collections.map_or_else(Vec::new, |collections| {
        collections
            .split(',')
            .map(str::trim)
            .filter(|collection| !collection.is_empty())
            .map(ToString::to_string)
            .collect()
    })
}

/// Returns a bounded search limit.
fn search_limit(limit: Option<usize>) -> Result<usize, ServiceError> {
    let limit = limit.unwrap_or_else(|| IndexSearchQuery::default().limit);
    if (1..=MAX_SEARCH_LIMIT).contains(&limit) {
        return Ok(limit);
    }

    Err(ServiceError::bad_request(format!(
        "limit must be between 1 and {MAX_SEARCH_LIMIT}",
    )))
}

#[cfg(test)]
mod tests;
