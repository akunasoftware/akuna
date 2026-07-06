//! Embedding bindings.

use akuna_core::embedding as core_embedding;

/// Embedding adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum EmbeddingError {
    /// Embedding runtime failure.
    #[error("{message}")]
    Runtime {
        /// Human-readable error message.
        message: String,
    },
}

/// Supported embedding model checkpoints.
#[derive(uniffi::Enum)]
pub enum EmbeddingModel {
    /// `sentence-transformers/all-MiniLM-L6-v2`.
    MiniLmL6,
    /// `sentence-transformers/all-MiniLM-L12-v2`.
    MiniLmL12,
    /// `BAAI/bge-small-en-v1.5`.
    BgeSmallEnV15,
    /// `BAAI/bge-base-en-v1.5`.
    BgeBaseEnV15,
    /// `BAAI/bge-large-en-v1.5`.
    BgeLargeEnV15,
    /// `sentence-transformers/all-mpnet-base-v2`.
    AllMpnetBaseV2,
    /// `BAAI/bge-m3` dense embeddings.
    BgeM3,
}

/// Construction options for [`TextEmbedder`].
#[derive(uniffi::Record)]
pub struct TextEmbedderOptions {
    /// Which embedding checkpoint to load.
    pub model: EmbeddingModel,
}

/// Text embedding model.
#[derive(uniffi::Object)]
pub struct TextEmbedder {
    inner: core_embedding::TextEmbedder,
}

#[uniffi::export(async_runtime = "tokio")]
/// Loads an embedding model.
pub async fn load_text_embedder(
    options: Option<TextEmbedderOptions>,
) -> Result<TextEmbedder, EmbeddingError> {
    let inner = core_embedding::TextEmbedder::new(core_options(options))
        .await
        .map_err(to_error)?;
    Ok(TextEmbedder { inner })
}

#[uniffi::export]
impl TextEmbedder {
    /// Embeds one document with an optional prompt.
    pub fn embed(
        &self,
        document: String,
        prompt: Option<String>,
    ) -> Result<Vec<f32>, EmbeddingError> {
        self.inner
            .embed_with_prompt(document, prompt.as_deref())
            .map_err(to_error)
    }

    /// Embeds documents in batches with an optional prompt.
    pub fn embed_batch(
        &self,
        documents: Vec<String>,
        batch_size: Option<u32>,
        prompt: Option<String>,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.inner
            .embed_batch_with_prompt(
                &documents,
                batch_size
                    .map(usize::try_from)
                    .transpose()
                    .map_err(to_error)?,
                prompt.as_deref(),
            )
            .map_err(to_error)
    }
}

// Keep binding defaults aligned with core defaults.
fn core_options(
    options: Option<TextEmbedderOptions>,
) -> core_embedding::TextEmbedderOptions {
    options.map_or_else(
        core_embedding::TextEmbedderOptions::default,
        |options| core_embedding::TextEmbedderOptions {
            model: options.model.into(),
            cache_dir: None,
        },
    )
}

impl From<EmbeddingModel> for core_embedding::EmbeddingModel {
    fn from(model: EmbeddingModel) -> Self {
        match model {
            EmbeddingModel::MiniLmL6 => Self::MiniLmL6,
            EmbeddingModel::MiniLmL12 => Self::MiniLmL12,
            EmbeddingModel::BgeSmallEnV15 => Self::BgeSmallEnV15,
            EmbeddingModel::BgeBaseEnV15 => Self::BgeBaseEnV15,
            EmbeddingModel::BgeLargeEnV15 => Self::BgeLargeEnV15,
            EmbeddingModel::AllMpnetBaseV2 => Self::AllMpnetBaseV2,
            EmbeddingModel::BgeM3 => Self::BgeM3,
        }
    }
}

// Keep binding errors string-only for language portability.
fn to_error(error: impl ToString) -> EmbeddingError {
    EmbeddingError::Runtime {
        message: error.to_string(),
    }
}
