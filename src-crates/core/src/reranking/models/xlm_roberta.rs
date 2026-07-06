use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{Bool, Int, Tensor, backend::Backend};
use burn_store::{
    KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::Deserialize;
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

use crate::ml::transformer::{BertEncoder, EncoderConfig, bert_encoder_remap};
use crate::ml::{cls_pooling, download_hf_model_files};

type TokenizedPairs<B> = (Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 2, Int>);

#[derive(Debug, Clone, Deserialize)]
struct XlmRobertaConfig {
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
struct XlmRobertaModel<B: Backend> {
    embeddings: XlmRobertaEmbeddings<B>,
    encoder: BertEncoder<B>,
}

/// Two-layer classification head producing a single relevance logit per input.
#[derive(Module, Debug)]
pub(crate) struct SequenceClassificationHead<B: Backend> {
    /// Hidden projection applied before the activation.
    pub(crate) dense: Linear<B>,
    /// Final projection producing one scalar logit per input.
    pub(crate) out_proj: Linear<B>,
}

#[derive(Module, Debug)]
pub(crate) struct XlmRobertaForSequenceClassification<B: Backend> {
    bert: XlmRobertaModel<B>,
    classifier: SequenceClassificationHead<B>,
}

/// Loaded XLM-RoBERTa reranker model ready for inference.
#[derive(Debug)]
pub(crate) struct XlmRobertaRerankerModel<B: Backend> {
    pub(crate) model: XlmRobertaForSequenceClassification<B>,
    tokenizer: Tokenizer,
}

impl XlmRobertaConfig {
    fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        crate::ml::load_json_config(path.as_ref(), "reranker config")
    }

    fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> XlmRobertaForSequenceClassification<B> {
        XlmRobertaForSequenceClassification {
            bert: XlmRobertaModel {
                embeddings: XlmRobertaEmbeddings::new(self, device),
                encoder: BertEncoder::init(
                    &EncoderConfig {
                        d_model: self.hidden_size,
                        d_ff: self.intermediate_size,
                        n_heads: self.num_attention_heads,
                        n_layers: self.num_hidden_layers,
                        layer_norm_eps: self.layer_norm_eps,
                    },
                    device,
                ),
            },
            classifier: SequenceClassificationHead {
                dense: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .init(device),
                out_proj: LinearConfig::new(self.hidden_size, 1).init(device),
            },
        }
    }
}

impl<B: Backend> XlmRobertaEmbeddings<B> {
    fn new(config: &XlmRobertaConfig, device: &B::Device) -> Self {
        Self {
            word_embeddings: EmbeddingConfig::new(
                config.vocab_size,
                config.hidden_size,
            )
            .init(device),
            position_embeddings: EmbeddingConfig::new(
                config.max_position_embeddings,
                config.hidden_size,
            )
            .init(device),
            token_type_embeddings: EmbeddingConfig::new(
                config.type_vocab_size,
                config.hidden_size,
            )
            .init(device),
            layer_norm: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            dropout: DropoutConfig::new(0.0).init(),
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
            Tensor::<B, 1, Int>::arange(2..(seq_len as i64 + 2), &device)
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

        self.dropout.forward(self.layer_norm.forward(embeddings))
    }
}

impl<B: Backend> XlmRobertaModel<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> XlmRobertaOutput<B> {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);
        let device = attention_mask.device();
        let zeros = Tensor::<B, 2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal(zeros);

        XlmRobertaOutput {
            hidden_states: self.encoder.forward(embeddings, mask_pad),
        }
    }
}

impl<B: Backend> XlmRobertaForSequenceClassification<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 1> {
        let output =
            self.bert.forward(input_ids, attention_mask, token_type_ids);
        let pooled = cls_pooling(output.hidden_states);
        let logits = self.classifier.forward(pooled);
        let [batch_size, _] = logits.dims();

        logits.reshape([batch_size])
    }
}

impl<B: Backend> SequenceClassificationHead<B> {
    pub(crate) fn forward(&self, pooled: Tensor<B, 2>) -> Tensor<B, 2> {
        self.out_proj.forward(self.dense.forward(pooled).tanh())
    }
}

impl<B: Backend> XlmRobertaRerankerModel<B> {
    /// Scores query/document `pairs`, returning one relevance logit per pair.
    pub(crate) fn score(
        &self,
        pairs: &[(&str, &str)],
        device: &B::Device,
    ) -> Result<Tensor<B, 1>> {
        let (input_ids, attention_mask, token_type_ids) =
            tokenize_pairs(&self.tokenizer, pairs, device)?;

        Ok(self
            .model
            .forward(input_ids, attention_mask, Some(token_type_ids)))
    }
}

/// Loads an XLM-RoBERTa reranker model.
pub(crate) async fn load_pretrained_xlm_roberta_reranker<B>(
    device: &B::Device,
    repo_id: &str,
    cache_dir: Option<PathBuf>,
) -> Result<XlmRobertaRerankerModel<B>>
where
    B: Backend,
{
    let files = download_hf_model_files(
        repo_id,
        "model.safetensors",
        cache_dir,
        "reranker model",
    )
    .await?;
    let config = XlmRobertaConfig::load_from_hf(&files.config_path)?;
    let mut model = config.init(device);
    load_pretrained_weights(&mut model, &files.weights_path)?;
    let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!(
                "failed to load reranker tokenizer from {}",
                files.tokenizer_path.display()
            )
        })?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: config.max_position_embeddings - 2,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(XlmRobertaRerankerModel { model, tokenizer })
}

fn load_pretrained_weights<B: Backend>(
    model: &mut XlmRobertaForSequenceClassification<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let mut key_mappings = vec![("^roberta\\.(.+)", "bert.$1")];
    key_mappings.extend(bert_encoder_remap());
    key_mappings.push((
        "bert\\.embeddings\\.LayerNorm",
        "bert.embeddings.layer_norm",
    ));
    let remapper = KeyRemapper::from_patterns(key_mappings)
        .context("failed to create reranker weight remapper")?;
    let mut store = SafetensorsStore::from_file(checkpoint_path.as_ref())
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load reranker weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

fn tokenize_pairs<B: Backend>(
    tokenizer: &Tokenizer,
    pairs: &[(&str, &str)],
    device: &B::Device,
) -> Result<TokenizedPairs<B>> {
    let inputs = pairs
        .iter()
        .map(|(query, document)| {
            EncodeInput::Dual((*query).into(), (*document).into())
        })
        .collect::<Vec<_>>();
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize reranker input batch")?;
    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1);
    let batch_size = pairs.len();
    let mut input_ids = vec![0i32; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];
    let mut token_type_ids = vec![0i32; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        for (token_index, token_id) in encoding.get_ids().iter().enumerate() {
            let position = batch_index * max_len + token_index;
            input_ids[position] = *token_id as i32;
            attention_mask[position] =
                encoding.get_attention_mask()[token_index] as f32;
            token_type_ids[position] =
                encoding.get_type_ids()[token_index] as i32;
        }
    }

    Ok((
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device)
            .reshape([batch_size, max_len]),
        Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
            .reshape([batch_size, max_len]),
        Tensor::<B, 1, Int>::from_ints(token_type_ids.as_slice(), device)
            .reshape([batch_size, max_len]),
    ))
}
