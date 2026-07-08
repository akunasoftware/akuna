/// Embedding failure.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Model assets or weights failed to load.
    #[error("embedding model load failed")]
    Load {
        /// Underlying model error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Embedding inference failed.
    #[error("embedding inference failed")]
    Inference {
        /// Underlying inference error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl EmbeddingError {
    /// Preserves an internal error from model loading.
    pub(crate) fn load(source: anyhow::Error) -> Self {
        Self::Load {
            source: crate::ml::boxed_model_error(source),
        }
    }

    /// Preserves an internal error from embedding inference.
    pub(crate) fn inference(source: anyhow::Error) -> Self {
        Self::Inference {
            source: crate::ml::boxed_model_error(source),
        }
    }
}
