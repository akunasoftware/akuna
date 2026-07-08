//! Local HTTP REST API server.

use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    http::{HeaderValue, Method, header::HOST},
};
use const_format::concatcp;
use tokio::net::TcpListener;
use tower_http::cors::{AllowOrigin, CorsLayer};
use utoipa::OpenApi;

use akuna_core::index::{
    IndexSearchResult, MetadataFilter, MetadataValue, Record,
    RecordRelationship,
};

use crate::api::{error::ApiErrorBody, knowledge};

const API_ADDRESS: &str = "127.0.0.1:9876";
const API_BASE_PATH: &str = "/api/v1";
const API_SERVER: &str = concatcp!("http://localhost:9876", API_BASE_PATH);
/// File name for generated OpenAPI JSON artifact.
pub(crate) const OPENAPI_FILE_NAME: &str = "openapi.json";

#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        knowledge::upsert_records,
        knowledge::read_record,
        knowledge::delete_record,
        knowledge::search_records,
    ),
    components(schemas(
        ApiErrorBody,
        IndexSearchResult,
        MetadataFilter,
        MetadataValue,
        Record,
        RecordRelationship,
    )),
    servers((url = API_SERVER))
)]
struct ApiDoc;

/// Runs the local REST API server.
pub async fn run() -> Result<()> {
    let address: SocketAddr = API_ADDRESS.parse()?;
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("Failed to bind API address {address}"))?;
    let openapi = ApiDoc::openapi();
    let api = knowledge::router()
        .await
        .map_err(|error| {
            anyhow::anyhow!("Failed to initialize knowledge API: {error:?}")
        })?
        .route(
            "/openapi.json",
            axum::routing::get(|| async { Json(openapi) }),
        )
        .layer(cors_layer());
    let app = Router::new().nest(API_BASE_PATH, api);

    tracing::info!("serving REST API at http://{address}");
    axum::serve(listener, app)
        .await
        .context("REST API server failed")
}

/// Generates and writes OpenAPI JSON schema.
pub(crate) fn generate_schema(out_dir: &std::path::Path) -> Result<PathBuf> {
    let schema_path = out_dir.join(OPENAPI_FILE_NAME);
    let schema_json = serde_json::to_string_pretty(&ApiDoc::openapi())
        .context("Failed to serialize OpenAPI schema")?;

    std::fs::write(&schema_path, schema_json).with_context(|| {
        format!("Failed to write {}", schema_path.display())
    })?;

    Ok(schema_path)
}

/// Builds CORS policy for same-host browser clients on any port.
fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, request| {
            let Some(host) = request.headers.get(HOST) else {
                return false;
            };

            is_same_host_origin(origin, host)
        }))
        .allow_methods([
            Method::DELETE,
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::OPTIONS,
        ])
        .allow_headers(tower_http::cors::Any)
}

/// Checks whether Origin host matches request Host, ignoring port.
fn is_same_host_origin(
    origin: &HeaderValue,
    request_host: &HeaderValue,
) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Ok(request_host) = request_host.to_str() else {
        return false;
    };
    let Some(host) = origin_host(origin) else {
        return false;
    };
    let Some(request_host) = authority_host(request_host) else {
        return false;
    };

    host == request_host
}

/// Extracts host from browser Origin header value.
fn origin_host(origin: &str) -> Option<&str> {
    let (_, authority) = origin.split_once("://")?;
    authority_host(authority)
}

/// Extracts host from authority string, preserving bracketed IPv6 hosts.
fn authority_host(authority: &str) -> Option<&str> {
    let authority = authority.split('/').next().unwrap_or(authority);

    if authority.starts_with('[') {
        return authority
            .split(']')
            .next()
            .map(|host| &authority[..=host.len()]);
    }

    authority.split(':').next()
}
