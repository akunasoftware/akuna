/// Reranking failure.
#[derive(Debug, thiserror::Error)]
pub enum RerankingError {
    /// Model assets or weights failed to load.
    #[error("reranking model load failed")]
    Load {
        /// Underlying model error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Reranking inference failed.
    #[error("reranking inference failed")]
    Inference {
        /// Underlying inference error.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl RerankingError {
    /// Preserves an internal error from model loading.
    pub(crate) fn load(source: anyhow::Error) -> Self {
        Self::Load {
            source: crate::ml::boxed_model_error(source),
        }
    }

    /// Preserves an internal error from reranking inference.
    pub(crate) fn inference(source: anyhow::Error) -> Self {
        Self::Inference {
            source: crate::ml::boxed_model_error(source),
        }
    }
}
