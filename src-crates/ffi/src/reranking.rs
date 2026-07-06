//! Reranking bindings.

use akuna_core::reranking as core_reranking;

/// Reranking adapter error.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum RerankingError {
    /// Reranking runtime failure.
    #[error("{message}")]
    Runtime {
        /// Human-readable error message.
        message: String,
    },
}

/// Supported reranker model checkpoints.
#[derive(uniffi::Enum)]
pub enum RerankingModel {
    /// `BAAI/bge-reranker-base`.
    BgeRerankerBase,
}

/// Construction options for [`TextReranker`].
#[derive(uniffi::Record)]
pub struct TextRerankerOptions {
    /// Which reranker checkpoint to load.
    pub model: RerankingModel,
}

/// Tunable behaviour for one rerank call.
#[derive(uniffi::Record)]
pub struct RerankOptions {
    /// Keep only the top `n` results when set.
    pub top_k: Option<u32>,
    /// Apply sigmoid normalization to scores.
    pub normalize: bool,
    /// Override the default inference batch size.
    pub batch_size: Option<u32>,
}

/// Query/document pair for scoring.
#[derive(uniffi::Record)]
pub struct TextPair {
    /// Query text.
    pub query: String,
    /// Document text.
    pub document: String,
}

/// Ranked document result.
#[derive(uniffi::Record)]
pub struct RerankResult {
    /// Original input index.
    pub index: u64,
    /// Document text.
    pub document: String,
    /// Relevance score.
    pub score: f32,
}

/// Cross-encoder text reranker.
#[derive(uniffi::Object)]
pub struct TextReranker {
    inner: core_reranking::TextReranker,
}

#[uniffi::export(async_runtime = "tokio")]
/// Loads a reranker model.
pub async fn load_text_reranker(
    options: Option<TextRerankerOptions>,
) -> Result<TextReranker, RerankingError> {
    let inner = core_reranking::TextReranker::new(core_options(options))
        .await
        .map_err(to_error)?;
    Ok(TextReranker { inner })
}

#[uniffi::export]
impl TextReranker {
    /// Scores query/document pairs in batches.
    pub fn score_pairs(
        &self,
        pairs: Vec<TextPair>,
        batch_size: Option<u32>,
    ) -> Result<Vec<f32>, RerankingError> {
        let pairs = pairs
            .into_iter()
            .map(|pair| (pair.query, pair.document))
            .collect::<Vec<_>>();
        self.inner
            .score_batch(
                &pairs,
                batch_size
                    .map(usize::try_from)
                    .transpose()
                    .map_err(to_error)?,
            )
            .map_err(to_error)
    }

    /// Ranks documents against a query.
    pub fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
        options: Option<RerankOptions>,
    ) -> Result<Vec<RerankResult>, RerankingError> {
        self.inner
            .rerank_with_options(
                query,
                &documents,
                core_rerank_options(options)?,
            )
            .map_err(to_error)?
            .into_iter()
            .map(|result| {
                Ok(RerankResult {
                    index: u64::try_from(result.index).map_err(to_error)?,
                    document: result.document,
                    score: result.score,
                })
            })
            .collect()
    }
}

// Keep binding defaults aligned with core defaults.
fn core_options(
    options: Option<TextRerankerOptions>,
) -> core_reranking::TextRerankerOptions {
    options.map_or_else(
        core_reranking::TextRerankerOptions::default,
        |options| core_reranking::TextRerankerOptions {
            model: options.model.into(),
            cache_dir: None,
        },
    )
}

// Keep binding call options aligned with core options.
fn core_rerank_options(
    options: Option<RerankOptions>,
) -> Result<core_reranking::RerankOptions, RerankingError> {
    let Some(options) = options else {
        return Ok(core_reranking::RerankOptions::default());
    };

    let mut core_options = core_reranking::RerankOptions::default();
    core_options.top_k = options
        .top_k
        .map(usize::try_from)
        .transpose()
        .map_err(to_error)?;
    core_options.normalize = options.normalize;
    core_options.batch_size = options
        .batch_size
        .map(usize::try_from)
        .transpose()
        .map_err(to_error)?;
    Ok(core_options)
}

impl From<RerankingModel> for core_reranking::RerankingModel {
    fn from(model: RerankingModel) -> Self {
        match model {
            RerankingModel::BgeRerankerBase => Self::BgeRerankerBase,
        }
    }
}

// Keep binding errors string-only for language portability.
fn to_error(error: impl ToString) -> RerankingError {
    RerankingError::Runtime {
        message: error.to_string(),
    }
}
