use akuna_core::index::IndexError;

use super::ServiceError;

#[test]
fn index_errors_map_by_variant() {
    let invalid = ServiceError::from(IndexError::InvalidInput {
        message: "search text must not be empty".to_string(),
    });
    let failed = ServiceError::from(IndexError::Open {
        source: Box::new(std::io::Error::other("manifest missing")),
    });

    assert!(matches!(invalid, ServiceError::BadRequest { .. }));
    assert!(matches!(failed, ServiceError::Internal { .. }));
}
