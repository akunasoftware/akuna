use serde::Serialize;

use crate::detection::vendor::content::ContentType;
use crate::detection::vendor::file::TypeInfo as VendorTypeInfo;

/// The source that resolved a file type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DetectionOrigin {
    /// A deterministic rule resolved the type without model inference.
    Rule,
    /// The bundled model resolved the type.
    Model,
}

/// Metadata describing a detected file type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct FileTypeInfo {
    /// Unique label for this file type.
    pub label: String,
    /// MIME type for this file type.
    pub mime_type: String,
    /// Broad file type group.
    pub group: String,
    /// Human-readable file type description.
    pub description: String,
    /// Known filename extensions.
    pub extensions: Vec<String>,
    /// Whether the file type is text-like.
    pub is_text: bool,
}

impl FileTypeInfo {
    /// Converts the vendored model metadata at the detection boundary.
    fn from_vendor(value: &VendorTypeInfo) -> Self {
        Self {
            label: value.label.to_owned(),
            mime_type: value.mime_type.to_owned(),
            group: value.group.to_owned(),
            description: value.description.to_owned(),
            extensions: value
                .extensions
                .iter()
                .map(|extension| (*extension).to_owned())
                .collect(),
            is_text: value.is_text,
        }
    }
}

/// A file type resolved from bytes or a file.
#[derive(Clone, Debug, PartialEq, Serialize, utoipa::ToSchema)]
pub struct FileType {
    info: FileTypeInfo,
    confidence: f32,
    origin: DetectionOrigin,
}

impl FileType {
    /// Returns metadata for the resolved type.
    pub fn info(&self) -> &FileTypeInfo {
        &self.info
    }

    /// Returns the resolution confidence from 0 to 1.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Returns whether a rule or model resolved the type.
    pub fn origin(&self) -> DetectionOrigin {
        self.origin
    }

    /// Returns the resolution confidence from 0 to 1.
    pub fn score(&self) -> f32 {
        self.confidence()
    }

    /// Builds a rule-resolved type from the vendored label metadata.
    pub(crate) fn ruled(content_type: ContentType) -> Self {
        Self::new(content_type, 1.0, DetectionOrigin::Rule)
    }

    /// Builds a model-resolved type from the vendored label metadata.
    pub(crate) fn inferred(content_type: ContentType, confidence: f32) -> Self {
        Self::new(content_type, confidence, DetectionOrigin::Model)
    }

    /// Converts one final vendored content type into the core result shape.
    fn new(
        content_type: ContentType,
        confidence: f32,
        origin: DetectionOrigin,
    ) -> Self {
        Self {
            info: FileTypeInfo::from_vendor(content_type.info()),
            confidence,
            origin,
        }
    }
}
