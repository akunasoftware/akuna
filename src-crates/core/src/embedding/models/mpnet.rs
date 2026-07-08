use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::Initializer::KaimingUniform;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{Int, Tensor, activation, backend::Backend};
use burn_store::{
    KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use crate::embedding::models::bert::{
    download_hf_model, mean_pooling, normalize_l2, prompt_sentences,
    tokenize_batch,
};

#[derive(Debug, Clone, Deserialize)]
struct MpnetConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    relative_attention_num_buckets: usize,
}

#[derive(Module, Debug)]
struct MpnetEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
pub(crate) struct MpnetModel<B: Backend> {
    embeddings: MpnetEmbeddings<B>,
    encoder: MpnetEncoder<B>,
}

#[derive(Module, Debug)]
struct MpnetEncoder<B: Backend> {
    relative_attention_bias: Embedding<B>,
    layers: Vec<MpnetLayer<B>>,
    num_attention_heads: usize,
    relative_attention_num_buckets: usize,
}

#[derive(Module, Debug)]
struct MpnetLayer<B: Backend> {
    attention: MpnetAttention<B>,
    intermediate: Linear<B>,
    output: Linear<B>,
    output_layer_norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct MpnetAttention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    o: Linear<B>,
    layer_norm: LayerNorm<B>,
}

/// Loaded MPNet embedding model ready for inference.
#[derive(Debug)]
pub(crate) struct MpnetEmbeddingModel<B: Backend> {
    pub(crate) model: MpnetModel<B>,
    tokenizer: Tokenizer,
}

impl MpnetConfig {
    fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        crate::ml::load_json_config(path.as_ref(), "embedding config")
    }

    fn init<B: Backend>(&self, device: &B::Device) -> MpnetModel<B> {
        MpnetModel {
            embeddings: MpnetEmbeddings::new(self, device),
            encoder: MpnetEncoder::new(self, device),
        }
    }
}

impl<B: Backend> MpnetEmbeddings<B> {
    fn new(config: &MpnetConfig, device: &B::Device) -> Self {
        let word_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device);
        let position_embeddings = EmbeddingConfig::new(
            config.max_position_embeddings,
            config.hidden_size,
        )
        .init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(0.0).init();

        Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
            dropout,
        }
    }

    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let word_embeddings = self.word_embeddings.forward(input_ids);
        let position_ids =
            Tensor::<B, 1, Int>::arange(2..(seq_len as i64 + 2), &device)
                .reshape([1, seq_len])
                .expand([batch_size, seq_len]);
        let position_embeddings =
            self.position_embeddings.forward(position_ids);
        let embeddings = word_embeddings + position_embeddings;
        let embeddings = self.layer_norm.forward(embeddings);

        self.dropout.forward(embeddings)
    }
}

impl<B: Backend> MpnetEncoder<B> {
    fn new(config: &MpnetConfig, device: &B::Device) -> Self {
        let relative_attention_bias = EmbeddingConfig::new(
            config.relative_attention_num_buckets,
            config.num_attention_heads,
        )
        .init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|_| MpnetLayer::new(config, device))
            .collect::<Vec<_>>();

        Self {
            relative_attention_bias,
            layers,
            num_attention_heads: config.num_attention_heads,
            relative_attention_num_buckets: config
                .relative_attention_num_buckets,
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = attention_mask.dims();
        let device = attention_mask.device();
        let buckets = relative_position_bucket_ids(
            seq_len,
            self.relative_attention_num_buckets,
        );
        let buckets =
            Tensor::<B, 1, Int>::from_ints(buckets.as_slice(), &device)
                .reshape([seq_len, seq_len]);
        let position_bias = self
            .relative_attention_bias
            .forward(buckets)
            .swap_dims(1, 2)
            .swap_dims(0, 1)
            .reshape([1, self.num_attention_heads, seq_len, seq_len]);
        let attention_mask = attention_mask
            .reshape([batch_size, 1, 1, seq_len])
            .expand([batch_size, self.num_attention_heads, seq_len, seq_len]);
        let ones = Tensor::<B, 4>::ones(attention_mask.shape(), &device);
        let attention_bias = (ones - attention_mask) * -10000.0 + position_bias;

        self.layers
            .iter()
            .fold(hidden_states, |hidden_states, layer| {
                layer.forward(hidden_states, attention_bias.clone())
            })
    }
}

impl<B: Backend> MpnetLayer<B> {
    fn new(config: &MpnetConfig, device: &B::Device) -> Self {
        let attention = MpnetAttention::new(config, device);
        let intermediate =
            mpnet_linear(config.hidden_size, config.intermediate_size, device);
        let output =
            mpnet_linear(config.intermediate_size, config.hidden_size, device);
        let output_layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        Self {
            attention,
            intermediate,
            output,
            output_layer_norm,
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let attention_output =
            self.attention.forward(hidden_states, attention_bias);
        let output = activation::gelu(
            self.intermediate.forward(attention_output.clone()),
        );
        let output = self.output.forward(output);

        self.output_layer_norm.forward(output + attention_output)
    }
}

impl<B: Backend> MpnetAttention<B> {
    fn new(config: &MpnetConfig, device: &B::Device) -> Self {
        let q = mpnet_linear(config.hidden_size, config.hidden_size, device);
        let k = mpnet_linear(config.hidden_size, config.hidden_size, device);
        let v = mpnet_linear(config.hidden_size, config.hidden_size, device);
        let o = mpnet_linear(config.hidden_size, config.hidden_size, device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);

        Self {
            q,
            k,
            v,
            o,
            layer_norm,
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let num_heads = attention_bias.dims()[1];
        let head_size = hidden_size / num_heads;
        let query = self
            .q
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, num_heads, head_size])
            .swap_dims(1, 2);
        let key = self
            .k
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, num_heads, head_size])
            .swap_dims(1, 2);
        let value = self
            .v
            .forward(hidden_states.clone())
            .reshape([batch_size, seq_len, num_heads, head_size])
            .swap_dims(1, 2);
        let attention_scores = query.matmul(key.swap_dims(2, 3))
            / (head_size as f64).sqrt()
            + attention_bias;
        let attention_probs = activation::softmax(attention_scores, 3);
        let context = attention_probs.matmul(value).swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            hidden_size,
        ]);
        let attention_output = self.o.forward(context);

        self.layer_norm.forward(attention_output + hidden_states)
    }
}

impl<B: Backend> MpnetModel<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let embeddings = self.embeddings.forward(input_ids);
        self.encoder.forward(embeddings, attention_mask)
    }
}

impl<B> MpnetEmbeddingModel<B>
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
        let (input_ids, attention_mask) = tokenize_batch::<B>(
            &self.tokenizer,
            &prompted_sentence_refs,
            device,
        )?;
        let hidden_states =
            self.model.forward(input_ids, attention_mask.clone());
        let embeddings = mean_pooling(hidden_states, attention_mask);

        Ok(normalize_l2(embeddings))
    }
}

/// Loads an MPNet embedding model.
pub(crate) async fn load_pretrained_mpnet_embedding<B>(
    device: &B::Device,
    repo_id: &str,
    max_length: usize,
    cache_dir: Option<PathBuf>,
) -> Result<MpnetEmbeddingModel<B>>
where
    B: Backend,
{
    let files = download_hf_model(repo_id, cache_dir).await?;
    let config = MpnetConfig::load_from_hf(&files.config_path)?;
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
    let max_length =
        max_length.min(config.max_position_embeddings.saturating_sub(2));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(MpnetEmbeddingModel { model, tokenizer })
}

fn load_pretrained_weights<B: Backend>(
    model: &mut MpnetModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let key_mappings = vec![
        ("^mpnet\\.(.+)", "$1"),
        ("encoder\\.layer\\.([0-9]+)", "encoder.layers.$1"),
        ("attention\\.attn\\.q", "attention.q"),
        ("attention\\.attn\\.k", "attention.k"),
        ("attention\\.attn\\.v", "attention.v"),
        ("attention\\.attn\\.o", "attention.o"),
        ("attention\\.LayerNorm", "attention.layer_norm"),
        (
            "(layers\\.[0-9]+)\\.intermediate\\.dense",
            "$1.intermediate",
        ),
        ("(layers\\.[0-9]+)\\.output\\.dense", "$1.output"),
        (
            "(layers\\.[0-9]+)\\.output\\.LayerNorm",
            "$1.output_layer_norm",
        ),
        ("embeddings\\.LayerNorm", "embeddings.layer_norm"),
    ];

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

fn mpnet_linear<B: Backend>(
    input_size: usize,
    output_size: usize,
    device: &B::Device,
) -> Linear<B> {
    LinearConfig::new(input_size, output_size)
        .with_initializer(KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: false,
        })
        .init(device)
}

fn relative_position_bucket_ids(
    seq_len: usize,
    num_buckets: usize,
) -> Vec<i32> {
    let max_distance = 128;
    (0..seq_len)
        .flat_map(|query_position| {
            (0..seq_len).map(move |key_position| {
                relative_position_bucket(
                    key_position as i32 - query_position as i32,
                    num_buckets as i32,
                    max_distance,
                )
            })
        })
        .collect()
}

fn relative_position_bucket(
    relative_position: i32,
    num_buckets: i32,
    max_distance: i32,
) -> i32 {
    let half_buckets = num_buckets / 2;
    let mut bucket = if relative_position > 0 {
        half_buckets
    } else {
        0
    };
    let relative_position = relative_position.abs();
    let max_exact = half_buckets / 2;

    if relative_position < max_exact {
        return bucket + relative_position;
    }

    let relative_position = relative_position.max(1) as f64;
    let max_exact_f64 = max_exact as f64;
    let max_distance_f64 = max_distance as f64;
    let relative_bucket = max_exact
        + ((relative_position / max_exact_f64).ln()
            / (max_distance_f64 / max_exact_f64).ln()
            * (half_buckets - max_exact) as f64) as i32;

    bucket += relative_bucket.min(half_buckets - 1);
    bucket
}
