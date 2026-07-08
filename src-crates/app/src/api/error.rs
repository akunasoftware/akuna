//! Shared service and HTTP API errors.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use akuna_core::index::IndexError;

/// Error returned by service interfaces.
#[derive(Debug)]
pub enum ServiceError {
    /// Input was invalid.
    BadRequest { message: String },
    /// Requested entity was not found.
    NotFound { message: String },
    /// Operation failed unexpectedly.
    Internal { message: String },
}

impl ServiceError {
    /// Builds a bad request service error.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::BadRequest {
            message: message.into(),
        }
    }

    /// Builds a not found service error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound {
            message: message.into(),
        }
    }

    /// Builds an internal service error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

impl From<serde_json::Error> for ServiceError {
    fn from(source: serde_json::Error) -> Self {
        Self::bad_request(source.to_string())
    }
}

impl From<IndexError> for ServiceError {
    fn from(source: IndexError) -> Self {
        match source {
            IndexError::InvalidInput { message } => Self::bad_request(message),
            error => Self::internal(error.to_string()),
        }
    }
}

/// HTTP error body.
#[derive(Serialize, utoipa::ToSchema)]
pub struct ApiErrorBody {
    /// Machine-readable error code.
    pub error: String,
    /// Human-readable error message.
    pub message: String,
}

/// HTTP API error adapter.
pub struct ApiError {
    status: StatusCode,
    error: String,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorBody {
                error: self.error,
                message: self.message,
            }),
        )
            .into_response()
    }
}

impl From<ServiceError> for ApiError {
    fn from(source: ServiceError) -> Self {
        match source {
            ServiceError::BadRequest { message } => Self {
                status: StatusCode::BAD_REQUEST,
                error: "bad_request".to_string(),
                message,
            },
            ServiceError::NotFound { message } => Self {
                status: StatusCode::NOT_FOUND,
                error: "not_found".to_string(),
                message,
            },
            ServiceError::Internal { message } => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                error: "internal_error".to_string(),
                message,
            },
        }
    }
}

/// HTTP API result adapter.
pub type ApiResult<T> = Result<Json<T>, ApiError>;

#[cfg(test)]
#[path = "error/tests.rs"]
mod tests;
