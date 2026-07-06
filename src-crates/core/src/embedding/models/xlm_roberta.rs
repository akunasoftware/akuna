use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig,
};
use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use crate::ml::transformer::{BertEncoder, EncoderConfig, bert_encoder_remap};
use burn_store::{KeyRemapper, ModuleSnapshot, PytorchStore};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use crate::embedding::models::bert::{
    download_hf_model_with_weights, normalize_l2, prompt_sentences,
};
use crate::ml::text::{cls_pooling, xlm_roberta_position_ids};

type TokenizedPairs<B> = (Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 2, Int>);

#[derive(Debug, Clone, Deserialize)]
struct XlmRobertaConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    #[serde(default = "default_type_vocab_size")]
    type_vocab_size: usize,
    #[serde(default = "default_pad_token_id")]
    pad_token_id: i32,
}

#[derive(Debug)]
struct XlmRobertaOutput<B: Backend> {
    hidden_states: Tensor<B, 3>,
}

#[derive(Module, Debug)]
struct XlmRobertaEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
pub(crate) struct XlmRobertaModel<B: Backend> {
    embeddings: XlmRobertaEmbeddings<B>,
    encoder: BertEncoder<B>,
}

/// Loaded XLM-RoBERTa embedding model ready for inference.
#[derive(Debug)]
pub(crate) struct XlmRobertaEmbeddingModel<B: Backend> {
    pub(crate) model: XlmRobertaModel<B>,
    tokenizer: Tokenizer,
    max_length: usize,
    pad_token_id: i32,
}

impl XlmRobertaConfig {
    fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        crate::ml::text::load_json_config(path.as_ref(), "embedding config")
    }

    fn init<B: Backend>(&self, device: &B::Device) -> XlmRobertaModel<B> {
        let embeddings = XlmRobertaEmbeddings::new(self, device);
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

        XlmRobertaModel {
            embeddings,
            encoder,
        }
    }
}

impl<B: Backend> XlmRobertaEmbeddings<B> {
    fn new(config: &XlmRobertaConfig, device: &B::Device) -> Self {
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
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let word_embeddings = self.word_embeddings.forward(input_ids);
        let [batch_size, seq_len] = position_ids.dims();
        let device = position_ids.device();
        let position_embeddings =
            self.position_embeddings.forward(position_ids);
        let token_type_ids =
            Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device);
        let token_type_embeddings =
            self.token_type_embeddings.forward(token_type_ids);
        let embeddings =
            word_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(embeddings);

        self.dropout.forward(embeddings)
    }
}

impl<B: Backend> XlmRobertaModel<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        position_ids: Tensor<B, 2, Int>,
    ) -> XlmRobertaOutput<B> {
        let embeddings = self.embeddings.forward(input_ids, position_ids);
        let device = attention_mask.device();
        let zeros = Tensor::<B, 2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal(zeros);
        let hidden_states = self.encoder.forward(embeddings, mask_pad);

        XlmRobertaOutput { hidden_states }
    }
}

impl<B> XlmRobertaEmbeddingModel<B>
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
            .map(std::borrow::Cow::as_ref)
            .collect::<Vec<_>>();
        let (input_ids, attention_mask, position_ids) = tokenize_batch::<B>(
            &self.tokenizer,
            &prompted_sentence_refs,
            self.max_length,
            self.pad_token_id,
            device,
        )?;
        let output =
            self.model.forward(input_ids, attention_mask, position_ids);
        let embeddings = cls_pooling(output.hidden_states);

        Ok(normalize_l2(embeddings))
    }
}

/// Loads an XLM-RoBERTa embedding model.
pub(crate) async fn load_pretrained_xlm_roberta_embedding<B>(
    device: &B::Device,
    repo_id: &str,
    cache_dir: Option<PathBuf>,
) -> Result<XlmRobertaEmbeddingModel<B>>
where
    B: Backend,
{
    let files =
        download_hf_model_with_weights(repo_id, "pytorch_model.bin", cache_dir)
            .await?;
    let config = XlmRobertaConfig::load_from_hf(&files.config_path)?;
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
    let max_length = config.max_position_embeddings.saturating_sub(2);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(XlmRobertaEmbeddingModel {
        model,
        tokenizer,
        max_length,
        pad_token_id: config.pad_token_id,
    })
}

fn load_pretrained_weights<B: Backend>(
    model: &mut XlmRobertaModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let mut key_mappings =
        vec![("^roberta\\.(.+)", "$1"), ("^xlm_roberta\\.(.+)", "$1")];
    key_mappings.extend(bert_encoder_remap());
    key_mappings.push(("embeddings\\.LayerNorm", "embeddings.layer_norm"));

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .context("failed to create embedding weight remapper")?;
    let mut store = PytorchStore::from_file(checkpoint_path.as_ref())
        .map_indices_contiguous(false)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load embedding weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

fn tokenize_batch<B: Backend>(
    tokenizer: &Tokenizer,
    sentences: &[&str],
    max_length: usize,
    pad_token_id: i32,
    device: &B::Device,
) -> Result<TokenizedPairs<B>> {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize input batch")?;

    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1)
        .min(max_length);
    let batch_size = sentences.len();
    let mut input_ids = vec![pad_token_id; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        for token_index in 0..encoding.get_ids().len().min(max_len) {
            let offset = batch_index * max_len + token_index;
            let token_id = encoding.get_ids()[token_index] as i32;
            let mask = encoding.get_attention_mask()[token_index] as f32;
            input_ids[offset] = token_id;
            attention_mask[offset] = mask;
        }
    }

    let position_ids = xlm_roberta_position_ids(
        &attention_mask,
        batch_size,
        max_len,
        pad_token_id,
    );
    let input_ids =
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device)
            .reshape([batch_size, max_len]);
    let attention_mask =
        Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
            .reshape([batch_size, max_len]);
    let position_ids =
        Tensor::<B, 1, Int>::from_ints(position_ids.as_slice(), device)
            .reshape([batch_size, max_len]);

    Ok((input_ids, attention_mask, position_ids))
}

fn default_pad_token_id() -> i32 {
    1
}

fn default_type_vocab_size() -> usize {
    1
}
