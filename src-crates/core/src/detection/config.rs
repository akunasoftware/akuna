use std::borrow::Cow;

use crate::detection::vendor::content::ContentType;

/// Tunable model parameters for the Magika classifier.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of leading bytes used as features.
    pub beg_size: usize,
    /// Number of trailing bytes used as features.
    pub end_size: usize,
    /// Minimum input length before deep-learning inference is triggered.
    pub min_file_size_for_dl: usize,
    /// Token used to pad feature vectors to fixed length.
    pub padding_token: i32,
    /// Number of bytes consumed per padding block.
    pub block_size: usize,
    /// Per-class confidence thresholds for accepting predictions.
    pub thresholds: Cow<'static, [f32; ContentType::SIZE]>,
    /// Maps inferred labels to overridden output content types.
    pub overwrite_map: Cow<'static, [ContentType; ContentType::SIZE]>,
}

impl ModelConfig {
    /// Total feature vector length (`beg_size + end_size`).
    pub(crate) fn features_size(&self) -> usize {
        self.beg_size + self.end_size
    }
}
