use anyhow::Result;
use burn_wgpu::Wgpu;
use tokio::sync::OnceCell;

pub use burn_embed::{EmbeddingModel, TextEmbeddingOptions};

/// Default backend for the shared text embedding model.
pub type DefaultEmbeddingBackend = Wgpu;

/// Default text embedding model type used by akuna.
pub type TextEmbedding = burn_embed::TextEmbedding<DefaultEmbeddingBackend>;

static MODEL: OnceCell<TextEmbedding> = OnceCell::const_new();

/// Returns the shared default text embedding model, initializing it once on first use.
pub async fn model() -> Result<&'static TextEmbedding> {
    MODEL
        .get_or_try_init(|| async {
            let device = burn_wgpu::WgpuDevice::DefaultDevice;
            TextEmbedding::new_with_device(&device, Default::default()).await
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinSet;

    #[test]
    fn text_embedding_defaults_to_l12() {
        assert_eq!(
            TextEmbeddingOptions::default().model,
            EmbeddingModel::MiniLmL12
        );
    }

    #[tokio::test]
    async fn embeds_text_single_and_batch() {
        let model = model().await.expect("shared model should load");

        let embedding = model
            .embed("Hello world")
            .expect("single embed should work");
        assert!(!embedding.is_empty());

        let batch = model
            .embed_batch(&["Hello world", "Rust embeddings"], None)
            .expect("batch embed should work");
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|embedding| !embedding.is_empty()));
    }

    #[tokio::test]
    async fn embeds_text_batches_concurrently() {
        const TEXT_SAMPLE: &str = "the quick brown fox jumps over the lazy dog";
        const TEXT_SAMPLE_COUNT: usize = 10;
        const CONCURRENT_BATCHES: usize = 5;

        let expanded_texts = vec![TEXT_SAMPLE; TEXT_SAMPLE_COUNT];
        let model = model().await.expect("shared model should load");

        let mut tasks = JoinSet::new();
        for _ in 0..CONCURRENT_BATCHES {
            let texts = expanded_texts.clone();
            tasks.spawn(async move { model.embed_batch(&texts, None) });
        }

        let mut completed = 0;
        while let Some(result) = tasks.join_next().await {
            let embeddings = result
                .expect("embedding task should complete")
                .expect("batch should embed");

            assert_eq!(embeddings.len(), expanded_texts.len());
            assert!(embeddings.iter().all(|embedding| !embedding.is_empty()));
            completed += 1;
        }

        assert_eq!(completed, CONCURRENT_BATCHES);
    }
}
