use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig,
};
use burn::tensor::{Bool, Int, Tensor, backend::Backend};
use burn_store::{
    KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use crate::ml::text::{HfModelFiles, cls_pooling, download_hf_model_files};
use crate::ml::transformer::{BertEncoder, EncoderConfig, bert_encoder_remap};

/// How a sequence's token hidden states collapse to one embedding vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoolingStrategy {
    /// Mean over token hidden states.
    Mean,
    /// First token (CLS) hidden state.
    Cls,
}

#[derive(Debug, Clone, Deserialize)]
struct BertConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f64,
}

#[derive(Debug)]
struct BertOutput<B: Backend> {
    hidden_states: Tensor<B, 3>,
}

#[derive(Module, Debug)]
struct BertEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
pub(crate) struct BertModel<B: Backend> {
    embeddings: BertEmbeddings<B>,
    encoder: BertEncoder<B>,
}

/// Loaded BERT-family embedding model ready for inference.
#[derive(Debug)]
pub(crate) struct BertEmbeddingModel<B: Backend> {
    pub(crate) model: BertModel<B>,
    tokenizer: Tokenizer,
    pooling: PoolingStrategy,
}

impl BertConfig {
    /// Reads a BERT `config.json` file from disk.
    pub fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        crate::ml::text::load_json_config(path.as_ref(), "embedding config")
    }

    /// Initializes a BERT model with this config on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BertModel<B> {
        let embeddings = BertEmbeddings::new(self, device);
        let encoder = BertEncoder::init(
            &EncoderConfig {
                d_model: self.hidden_size,
                d_ff: self.intermediate_size,
                n_heads: self.num_attention_heads,
                n_layers: self.num_hidden_layers,
                layer_norm_eps: self.layer_norm_eps,
            },
            device,
        );

        BertModel {
            embeddings,
            encoder,
        }
    }
}

impl<B: Backend> BertEmbeddings<B> {
    fn new(config: &BertConfig, device: &B::Device) -> Self {
        let word_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device);
        let position_embeddings = EmbeddingConfig::new(
            config.max_position_embeddings,
            config.hidden_size,
        )
        .init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(config.type_vocab_size, config.hidden_size)
                .init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(0.0).init();

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let word_embeddings = self.word_embeddings.forward(input_ids);

        let position_ids =
            Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
                .reshape([1, seq_len])
                .expand([batch_size, seq_len]);
        let position_embeddings =
            self.position_embeddings.forward(position_ids);

        let token_type_ids = token_type_ids.unwrap_or_else(|| {
            Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device)
        });
        let token_type_embeddings =
            self.token_type_embeddings.forward(token_type_ids);

        let embeddings =
            word_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}

impl<B: Backend> BertModel<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> BertOutput<B> {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);
        let device = attention_mask.device();
        let zeros = Tensor::<B, 2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal(zeros);
        let hidden_states = self.encoder.forward(embeddings, mask_pad);

        BertOutput { hidden_states }
    }
}

impl<B> BertEmbeddingModel<B>
where
    B: Backend,
{
    /// Runs the model over a batch and returns L2-normalized embeddings.
    pub(crate) fn encode(
        &self,
        sentences: &[&str],
        prompt: Option<&str>,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let prompted_sentences = prompt_sentences(sentences, prompt);
        let prompted_sentence_refs = prompted_sentences
            .iter()
            .map(Cow::as_ref)
            .collect::<Vec<_>>();
        let (input_ids, attention_mask) = tokenize_batch::<B>(
            &self.tokenizer,
            &prompted_sentence_refs,
            device,
        )?;
        let output =
            self.model.forward(input_ids, attention_mask.clone(), None);

        let embeddings = match self.pooling {
            PoolingStrategy::Mean => {
                mean_pooling(output.hidden_states, attention_mask)
            }
            PoolingStrategy::Cls => cls_pooling(output.hidden_states),
        };

        Ok(normalize_l2(embeddings))
    }
}

/// Prefixes each sentence with `prompt` and trims surrounding whitespace.
pub(crate) fn prompt_sentences<'a>(
    sentences: &[&'a str],
    prompt: Option<&str>,
) -> Vec<Cow<'a, str>> {
    // SentenceTransformers strips input strings before tokenization.
    sentences
        .iter()
        .map(|sentence| match prompt {
            Some(prompt) => Cow::Owned(format!("{prompt}{}", sentence.trim())),
            None => Cow::Borrowed(sentence.trim()),
        })
        .collect()
}

/// Loads a BERT-family embedding model.
pub(crate) async fn load_pretrained_bert_embedding<B>(
    device: &B::Device,
    repo_id: &str,
    pooling: PoolingStrategy,
    max_length: Option<usize>,
    cache_dir: Option<PathBuf>,
) -> Result<BertEmbeddingModel<B>>
where
    B: Backend,
{
    let files = download_hf_model(repo_id, cache_dir).await?;
    let config = BertConfig::load_from_hf(&files.config_path)?;
    let mut model = config.init(device);
    load_pretrained_weights(&mut model, &files.weights_path)?;
    let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!(
                "failed to load embedding tokenizer from {}",
                files.tokenizer_path.display()
            )
        })?;
    let max_length = max_length
        .unwrap_or(config.max_position_embeddings)
        .min(config.max_position_embeddings);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(BertEmbeddingModel {
        model,
        tokenizer,
        pooling,
    })
}

/// Resolves default BERT-family model assets.
pub(crate) async fn download_hf_model(
    repo_id: &str,
    cache_dir: Option<PathBuf>,
) -> Result<HfModelFiles> {
    download_hf_model_with_weights(repo_id, "model.safetensors", cache_dir)
        .await
}

/// Resolves BERT-family model assets.
pub(crate) async fn download_hf_model_with_weights(
    repo_id: &str,
    weights_file: &str,
    cache_dir: Option<PathBuf>,
) -> Result<HfModelFiles> {
    download_hf_model_files(repo_id, weights_file, cache_dir, "embedding model")
        .await
}

fn load_pretrained_weights<B: Backend>(
    model: &mut BertModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let mut key_mappings = vec![("^bert\\.(.+)", "$1")];
    key_mappings.extend(bert_encoder_remap());
    key_mappings.push(("embeddings\\.LayerNorm", "embeddings.layer_norm"));

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .context("failed to create embedding weight remapper")?;
    let mut store = SafetensorsStore::from_file(checkpoint_path.as_ref())
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load embedding weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

/// Tokenizes a batch and returns `(input_ids, attention_mask)` tensors.
pub(crate) fn tokenize_batch<B: Backend>(
    tokenizer: &Tokenizer,
    sentences: &[&str],
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2>)> {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize input batch")?;

    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1);

    let batch_size = sentences.len();
    let mut input_ids = vec![0i32; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        for (token_index, token_id) in encoding.get_ids().iter().enumerate() {
            input_ids[batch_index * max_len + token_index] = *token_id as i32;
            attention_mask[batch_index * max_len + token_index] =
                encoding.get_attention_mask()[token_index] as f32;
        }
    }

    let input_ids =
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device)
            .reshape([batch_size, max_len]);
    let attention_mask =
        Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
            .reshape([batch_size, max_len]);

    Ok((input_ids, attention_mask))
}

/// Mean-pools token hidden states weighted by the attention mask.
pub(crate) fn mean_pooling<B: Backend>(
    hidden_states: Tensor<B, 3>,
    attention_mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();
    let mask_expanded = attention_mask
        .clone()
        .reshape([batch_size, seq_len, 1])
        .expand([batch_size, seq_len, hidden_size]);
    let sum_hidden = (hidden_states * mask_expanded)
        .sum_dim(1)
        .reshape([batch_size, hidden_size]);
    let token_counts = attention_mask
        .sum_dim(1)
        .reshape([batch_size, 1])
        .expand([batch_size, hidden_size])
        .clamp_min(1e-9);

    sum_hidden / token_counts
}

/// L2-normalizes each row of `embeddings`.
pub(crate) fn normalize_l2<B: Backend>(
    embeddings: Tensor<B, 2>,
) -> Tensor<B, 2> {
    use burn::tensor::linalg::{Norm, vector_normalize};

    vector_normalize(embeddings, Norm::L2, 1, 1e-12)
}
