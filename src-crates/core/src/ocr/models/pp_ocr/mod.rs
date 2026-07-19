//! PP-OCRv6 native model implementation.

pub(in crate::ocr) mod dictionary;
pub(in crate::ocr) mod native;
pub(in crate::ocr) mod postprocess;
pub(in crate::ocr) mod preprocess;
pub(in crate::ocr) mod runtime;
pub(in crate::ocr) mod spec;

#[cfg(test)]
mod tests;
