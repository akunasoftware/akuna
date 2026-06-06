//! Domain-specific file extractors.

pub(super) mod code;
pub(super) mod office;
pub(super) mod pdf;
pub(super) mod text;

#[cfg(feature = "ocr")]
pub(super) mod ocr;
