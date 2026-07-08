//! Record metadata and filtering shapes.
//!
//! ```
//! use akuna_core::metadata::{Metadata, MetadataFilter, MetadataValue};
//!
//! let metadata: Metadata = [("kind".to_string(), MetadataValue::Text("note".to_string()))]
//!     .into_iter()
//!     .collect();
//! assert!(MetadataFilter::Equals {
//!     key: "kind".to_string(),
//!     value: MetadataValue::Text("note".to_string()),
//! }
//! .matches(&metadata));
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Record metadata value.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum MetadataValue {
    /// Text metadata value.
    Text(String),
    /// Integer metadata value.
    Integer(i64),
    /// Float metadata value.
    Float(f64),
    /// Boolean metadata value.
    Boolean(bool),
}

/// Record metadata.
pub type Metadata = BTreeMap<String, MetadataValue>;

/// Record metadata predicate.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum MetadataFilter {
    /// Metadata key must equal the supplied value.
    Equals {
        /// Metadata key to inspect.
        key: String,
        /// Metadata value to compare.
        value: MetadataValue,
    },
    /// Every metadata predicate must match.
    All(Vec<MetadataFilter>),
}

impl MetadataFilter {
    /// Checks whether metadata satisfies this predicate.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Self::Equals { key, value } => metadata.get(key) == Some(value),
            Self::All(filters) => {
                filters.iter().all(|filter| filter.matches(metadata))
            }
        }
    }
}
