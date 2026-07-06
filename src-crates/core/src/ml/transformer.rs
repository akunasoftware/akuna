//! A BERT-style post-LayerNorm transformer encoder.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

use crate::ml::safe_matmul;

/// Mask fill value used for padded positions.
const MASK_FILL: f32 = -1.0e4;

/// Shared BERT encoder weight remap rules.
pub(crate) fn bert_encoder_remap() -> Vec<(&'static str, &'static str)> {
    vec![
        ("encoder\\.layer\\.([0-9]+)", "encoder.layers.$1"),
        ("attention\\.self\\.query", "mha.query"),
        ("attention\\.self\\.key", "mha.key"),
        ("attention\\.self\\.value", "mha.value"),
        ("attention\\.output\\.dense", "mha.output"),
        ("attention\\.output\\.LayerNorm", "norm_1"),
        ("intermediate\\.dense", "pwff.linear_inner"),
        ("(layers\\.[0-9]+)\\.output\\.dense", "$1.pwff.linear_outer"),
        ("(layers\\.[0-9]+)\\.output\\.LayerNorm", "$1.norm_2"),
    ]
}

/// Applies a burn `Linear` (`O = I·W + b`) to a `[batch, seq, d_in]` input.
fn linear_safe<B: Backend>(
    linear: &Linear<B>,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch, seq, d_in] = x.dims();
    let weight = linear.weight.val();
    let d_out = weight.dims()[1];
    let mut out = safe_matmul(x.reshape([batch * seq, d_in]), weight);
    if let Some(bias) = &linear.bias {
        out = out + bias.val().reshape([1, d_out]);
    }
    out.reshape([batch, seq, d_out])
}

/// Configuration for a [`BertEncoder`].
pub(crate) struct EncoderConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub layer_norm_eps: f64,
}

#[derive(Module, Debug)]
struct Mha<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    n_heads: usize,
}

impl<B: Backend> Mha<B> {
    fn init(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let linear = || LinearConfig::new(d_model, d_model).init(device);
        Self {
            query: linear(),
            key: linear(),
            value: linear(),
            output: linear(),
            n_heads,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();
        let head_dim = d_model / self.n_heads;
        let split = |t: Tensor<B, 3>| {
            t.reshape([batch, seq, self.n_heads, head_dim])
                .swap_dims(1, 2)
        };
        let q = split(linear_safe(&self.query, x.clone()));
        let k = split(linear_safe(&self.key, x.clone()));
        let v = split(linear_safe(&self.value, x));

        // [B, H, S, S]; K = head_dim (small, exact). Scale after, in f32.
        let scores = safe_matmul(q, k.swap_dims(2, 3))
            .div_scalar((head_dim as f32).sqrt());
        let scores =
            scores.mask_fill(mask_pad.reshape([batch, 1, 1, seq]), MASK_FILL);
        let weights = softmax(scores, 3);

        // [B, H, S, head_dim]; K = S (>=512 corrupts native matmul) -> chunked.
        let context = safe_matmul(weights, v)
            .swap_dims(1, 2)
            .reshape([batch, seq, d_model]);
        linear_safe(&self.output, context)
    }
}

#[derive(Module, Debug)]
struct Pwff<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
}

impl<B: Backend> Pwff<B> {
    fn init(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        Self {
            linear_inner: LinearConfig::new(d_model, d_ff).init(device),
            linear_outer: LinearConfig::new(d_ff, d_model).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        linear_safe(
            &self.linear_outer,
            gelu(linear_safe(&self.linear_inner, x)),
        )
    }
}

#[derive(Module, Debug)]
struct EncoderLayer<B: Backend> {
    mha: Mha<B>,
    pwff: Pwff<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn init(config: &EncoderConfig, device: &B::Device) -> Self {
        let norm = || {
            LayerNormConfig::new(config.d_model)
                .with_epsilon(config.layer_norm_eps)
                .init(device)
        };
        Self {
            mha: Mha::init(config.d_model, config.n_heads, device),
            pwff: Pwff::init(config.d_model, config.d_ff, device),
            norm_1: norm(),
            norm_2: norm(),
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        // Post-LN: norm_1 after the attention residual, norm_2 after the FFN.
        let x = self
            .norm_1
            .forward(x.clone() + self.mha.forward(x, mask_pad));
        self.norm_2.forward(x.clone() + self.pwff.forward(x))
    }
}

/// A post-LayerNorm BERT transformer encoder.
#[derive(Module, Debug)]
pub(crate) struct BertEncoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> BertEncoder<B> {
    pub(crate) fn init(config: &EncoderConfig, device: &B::Device) -> Self {
        Self {
            layers: (0..config.n_layers)
                .map(|_| EncoderLayer::init(config, device))
                .collect(),
        }
    }

    /// `x`: `[batch, seq, d_model]`; `mask_pad`: `[batch, seq]` true at padding.
    pub(crate) fn forward(
        &self,
        x: Tensor<B, 3>,
        mask_pad: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, mask_pad.clone());
        }
        x
    }
}
