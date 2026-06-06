//! Reusable native-burn building blocks for hand-written CNN / transformer
//! models loaded directly from HuggingFace `safetensors`.
//!
//! Shared by the native PP-DocLayout and PP-OCR implementations so each model
//! is a plain Rust module (no ONNX-generated code, no `.bpk`). Everything here
//! is backend-generic and loads weights from a [`SafeTensors`] archive.
#![allow(dead_code, clippy::too_many_arguments)]

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, ConvTranspose2d};
use burn::tensor::activation::softmax;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use safetensors::{Dtype, SafeTensors};

use crate::ml::safe_matmul;

// ---- raw safetensors readers ----------------------------------------------

/// Reads a tensor's raw f32 values, validating dtype and shape.
pub(crate) fn read_f32_values(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name}"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("tensor {name} must be float32");
    }
    if tensor.shape() != shape {
        bail!(
            "tensor {name} shape {:?} does not match expected {:?}",
            tensor.shape(),
            shape
        );
    }
    Ok(tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Reads a 1-D f32 tensor (BN params, biases, norm weights).
pub(crate) fn read_vec<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    name: &str,
    len: usize,
    device: &B::Device,
) -> Result<Tensor<B, 1>> {
    Ok(Tensor::from_data(
        TensorData::new(read_f32_values(tensors, name, &[len])?, [len]),
        device,
    ))
}

/// Reads a conv weight `[out, in/groups, kh, kw]` in native burn layout.
pub(crate) fn read_conv_weight<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    name: &str,
    out_ch: usize,
    in_per_group: usize,
    kernel: [usize; 2],
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let shape = [out_ch, in_per_group, kernel[0], kernel[1]];
    Ok(Tensor::from_data(
        TensorData::new(read_f32_values(tensors, name, &shape)?, shape),
        device,
    ))
}

/// Reads a 1-D conv weight stored as `[out, in/groups, k]` and returns it as a
/// 2-D conv weight `[out, in/groups, 1, k]` (the model applies these as height-1
/// Conv2d).
pub(crate) fn read_conv1d_as_conv2d_weight<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    name: &str,
    out_ch: usize,
    in_per_group: usize,
    kernel: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let values =
        read_f32_values(tensors, name, &[out_ch, in_per_group, kernel])?;
    Ok(Tensor::from_data(
        TensorData::new(values, [out_ch, in_per_group, 1, kernel]),
        device,
    ))
}

/// Reads a transposed-conv weight `[in, out/groups, kh, kw]` (burn layout —
/// matches the PaddlePaddle export, no transpose required).
pub(crate) fn read_conv_transpose_weight<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    name: &str,
    in_ch: usize,
    out_per_group: usize,
    kernel: [usize; 2],
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let shape = [in_ch, out_per_group, kernel[0], kernel[1]];
    Ok(Tensor::from_data(
        TensorData::new(read_f32_values(tensors, name, &shape)?, shape),
        device,
    ))
}

/// Reads a linear weight stored as `[out, in]` and returns it transposed to
/// `[in, out]` for `x @ w`.
pub(crate) fn read_linear_weight<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    name: &str,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let values = read_f32_values(tensors, name, &[out_features, in_features])?;
    let mut transposed = vec![0.0; values.len()];
    for o in 0..out_features {
        for i in 0..in_features {
            transposed[i * out_features + o] = values[o * in_features + i];
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(transposed, [in_features, out_features]),
        device,
    ))
}

// ---- activations -----------------------------------------------------------

/// Pointwise activation applied after a conv / linear.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Activation {
    Identity,
    Relu,
    Silu,
    /// Exact erf-based GELU.
    Gelu,
    Sigmoid,
    /// `clamp(alpha * x + beta, 0, 1)`.
    HardSigmoid {
        alpha: f64,
        beta: f64,
    },
    /// `x * clamp(x/6 + 0.5, 0, 1)`.
    HardSwish,
}

pub(crate) fn relu<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    x.clamp_min(0.0)
}

pub(crate) fn silu<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    x.clone() / ((x * -1.0).exp() + 1.0)
}

pub(crate) fn gelu<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    x.clone() * 0.5 * ((x / std::f64::consts::SQRT_2).erf() + 1.0)
}

pub(crate) fn sigmoid<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    ((x * -1.0).exp() + 1.0).recip()
}

pub(crate) fn hard_sigmoid<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
    alpha: f64,
    beta: f64,
) -> Tensor<B, D> {
    (x * alpha + beta).clamp(0.0, 1.0)
}

pub(crate) fn apply_activation<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
    act: Activation,
) -> Tensor<B, D> {
    match act {
        Activation::Identity => x,
        Activation::Relu => relu(x),
        Activation::Silu => silu(x),
        Activation::Gelu => gelu(x),
        Activation::Sigmoid => sigmoid(x),
        Activation::HardSigmoid { alpha, beta } => hard_sigmoid(x, alpha, beta),
        Activation::HardSwish => {
            let gate = hard_sigmoid(x.clone(), 0.166_666_701_436_042_79, 0.5);
            x * gate
        }
    }
}

/// Inference-mode batch norm over the channel (dim 1) axis of a `[N,C,H,W]`.
pub(crate) fn batch_norm_inference<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    running_mean: Tensor<B, 1>,
    running_var: Tensor<B, 1>,
    epsilon: f64,
) -> Tensor<B, 4> {
    let [_n, c, _h, _w] = x.dims();
    let shape = [1, c, 1, 1];
    (x - running_mean.reshape(shape))
        * (running_var.reshape(shape) + epsilon).sqrt().recip()
        * weight.reshape(shape)
        + bias.reshape(shape)
}

const BN_EPS: f64 = 1e-5;

/// Inference-mode BatchNorm2d over the channel axis, loaded from
/// `{prefix}.{weight,bias,running_mean,running_var}`.
#[derive(Debug)]
pub(crate) struct BatchNorm2dLayer<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    running_mean: Tensor<B, 1>,
    running_var: Tensor<B, 1>,
}

type BnParams<B> = BatchNorm2dLayer<B>;

impl<B: Backend<FloatElem = f32>> BatchNorm2dLayer<B> {
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: read_vec(
                tensors,
                &format!("{prefix}.weight"),
                channels,
                device,
            )?,
            bias: read_vec(
                tensors,
                &format!("{prefix}.bias"),
                channels,
                device,
            )?,
            running_mean: read_vec(
                tensors,
                &format!("{prefix}.running_mean"),
                channels,
                device,
            )?,
            running_var: read_vec(
                tensors,
                &format!("{prefix}.running_var"),
                channels,
                device,
            )?,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        batch_norm_inference(
            x,
            self.weight.clone(),
            self.bias.clone(),
            self.running_mean.clone(),
            self.running_var.clone(),
            BN_EPS,
        )
    }
}

// ---- Conv2d (+ optional BN) + activation -----------------------------------

/// A 2-D convolution with an optional folded BatchNorm and a pointwise
/// activation. Two storage forms are supported:
/// - `conv_bn`: `{conv}.weight` (no conv bias) + `{norm}.{weight,bias,running_mean,running_var}`.
/// - `conv_bias`: `{conv}.weight` + `{conv}.bias` (no norm).
#[derive(Debug)]
pub(crate) struct ConvLayer<B: Backend> {
    conv: Conv2d<B>,
    norm: Option<BnParams<B>>,
    activation: Activation,
}

/// Geometry of a conv: kernel, stride, padding (top,left,bottom,right), groups.
#[derive(Debug, Clone)]
pub(crate) struct ConvGeom {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padding: PaddingConfig2d,
    pub groups: usize,
}

impl ConvGeom {
    /// Symmetric `pad` on a `kernel` conv.
    pub fn k(kernel: usize, stride: usize, pad: usize, groups: usize) -> Self {
        Self {
            kernel: [kernel, kernel],
            stride: [stride, stride],
            padding: PaddingConfig2d::Explicit(pad, pad, pad, pad),
            groups,
        }
    }

    /// 1x1 pointwise (valid).
    pub fn pointwise() -> Self {
        Self {
            kernel: [1, 1],
            stride: [1, 1],
            padding: PaddingConfig2d::Valid,
            groups: 1,
        }
    }
}

impl<B: Backend<FloatElem = f32>> ConvLayer<B> {
    fn build_conv(
        tensors: &SafeTensors<'_>,
        weight_name: &str,
        bias_name: Option<&str>,
        in_ch: usize,
        out_ch: usize,
        geom: ConvGeom,
        device: &B::Device,
    ) -> Result<Conv2d<B>> {
        let bias = match bias_name {
            Some(name) => Some(Param::from_tensor(read_vec(
                tensors, name, out_ch, device,
            )?)),
            None => None,
        };
        Ok(Conv2d {
            weight: Param::from_tensor(read_conv_weight(
                tensors,
                weight_name,
                out_ch,
                in_ch / geom.groups,
                geom.kernel,
                device,
            )?),
            bias,
            stride: geom.stride,
            kernel_size: geom.kernel,
            dilation: [1, 1],
            groups: geom.groups,
            padding: geom.padding,
        })
    }

    /// Conv (no bias) + BatchNorm, from `{prefix}.convolution` / `{prefix}.{norm_suffix}`.
    pub(crate) fn conv_bn(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        norm_suffix: &str,
        in_ch: usize,
        out_ch: usize,
        geom: ConvGeom,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Self::build_conv(
                tensors,
                &format!("{prefix}.convolution.weight"),
                None,
                in_ch,
                out_ch,
                geom,
                device,
            )?,
            norm: Some(BnParams::load(
                tensors,
                &format!("{prefix}.{norm_suffix}"),
                out_ch,
                device,
            )?),
            activation,
        })
    }

    /// Weight-only height-1 Conv2d whose weight is stored as a 1-D conv
    /// `[out, in/groups, k]` (no bias, no BN), from `{prefix}.weight`.
    pub(crate) fn conv1d_weight_only(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        groups: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let pad = (kernel - 1) / 2;
        let conv = Conv2d {
            weight: Param::from_tensor(read_conv1d_as_conv2d_weight(
                tensors,
                &format!("{prefix}.weight"),
                out_ch,
                in_ch / groups,
                kernel,
                device,
            )?),
            bias: None,
            stride: [1, 1],
            kernel_size: [1, kernel],
            dilation: [1, 1],
            groups,
            padding: PaddingConfig2d::Explicit(0, pad, 0, pad),
        };
        Ok(Self {
            conv,
            norm: None,
            activation: Activation::Identity,
        })
    }

    /// Weight-only conv (no bias, no BN), from `{prefix}.weight`.
    pub(crate) fn conv_weight_only(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        geom: ConvGeom,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Self::build_conv(
                tensors,
                &format!("{prefix}.weight"),
                None,
                in_ch,
                out_ch,
                geom,
                device,
            )?,
            norm: None,
            activation,
        })
    }

    /// Biased conv (no BN), from `{prefix}.weight` / `{prefix}.bias`.
    pub(crate) fn conv_bias(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        geom: ConvGeom,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Self::build_conv(
                tensors,
                &format!("{prefix}.weight"),
                Some(&format!("{prefix}.bias")),
                in_ch,
                out_ch,
                geom,
                device,
            )?,
            norm: None,
            activation,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv.forward(x);
        let y = match &self.norm {
            Some(norm) => norm.forward(y),
            None => y,
        };
        apply_activation(y, self.activation)
    }
}

// ---- ConvTranspose2d (+ optional BN) + activation --------------------------

#[derive(Debug)]
pub(crate) struct ConvTransposeLayer<B: Backend> {
    conv: ConvTranspose2d<B>,
    norm: Option<BnParams<B>>,
    activation: Activation,
}

impl<B: Backend<FloatElem = f32>> ConvTransposeLayer<B> {
    /// Transposed conv with bias, `kernel=stride`, no padding (exact upsample).
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        weight_prefix: &str,
        bias_name: &str,
        norm: Option<&str>,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Result<Self> {
        let conv = ConvTranspose2d {
            weight: Param::from_tensor(read_conv_transpose_weight(
                tensors,
                &format!("{weight_prefix}.weight"),
                in_ch,
                out_ch,
                [kernel, kernel],
                device,
            )?),
            bias: Some(Param::from_tensor(read_vec(
                tensors, bias_name, out_ch, device,
            )?)),
            stride: [stride, stride],
            kernel_size: [kernel, kernel],
            dilation: [1, 1],
            groups: 1,
            padding: [0, 0],
            padding_out: [0, 0],
            channels: [in_ch, out_ch],
        };
        let norm = match norm {
            Some(prefix) => {
                Some(BnParams::load(tensors, prefix, out_ch, device)?)
            }
            None => None,
        };
        Ok(Self {
            conv,
            norm,
            activation,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.conv.forward(x);
        let y = match &self.norm {
            Some(norm) => norm.forward(y),
            None => y,
        };
        apply_activation(y, self.activation)
    }
}

// ---- Linear ----------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct LinearLayer<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    out_features: usize,
}

impl<B: Backend<FloatElem = f32>> LinearLayer<B> {
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        in_features: usize,
        out_features: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: read_linear_weight(
                tensors,
                &format!("{prefix}.weight"),
                in_features,
                out_features,
                device,
            )?,
            bias: read_vec(
                tensors,
                &format!("{prefix}.bias"),
                out_features,
                device,
            )?,
            out_features,
        })
    }

    /// `[B, S, in] -> [B, S, out]`.
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, in_features] = x.dims();
        let flat = x.reshape([batch * seq, in_features]);
        let projected = safe_matmul(flat, self.weight.clone())
            + self.bias.clone().unsqueeze();
        projected.reshape([batch, seq, self.out_features])
    }
}

// ---- LayerNorm (manual, over last dim) -------------------------------------

#[derive(Debug)]
pub(crate) struct LayerNormLayer<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    epsilon: f64,
}

impl<B: Backend<FloatElem = f32>> LayerNormLayer<B> {
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        dim: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: read_vec(
                tensors,
                &format!("{prefix}.weight"),
                dim,
                device,
            )?,
            bias: read_vec(tensors, &format!("{prefix}.bias"), dim, device)?,
            epsilon,
        })
    }

    /// Normalizes over the last (`dim`) axis of a `[B, S, dim]` tensor.
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, _s, dim] = x.dims();
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered
            * (var + self.epsilon).sqrt().recip()
            * self.weight.clone().reshape([1, 1, dim])
            + self.bias.clone().reshape([1, 1, dim])
    }
}

/// Scaled dot-product attention over `[B, heads, seq, head_dim]` tensors with a
/// fixed `scale`. Softmax over the key axis. Uses [`safe_matmul`] for the large
/// contraction dims.
pub(crate) fn scaled_dot_product_attention<B: Backend<FloatElem = f32>>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    scale: f64,
) -> Tensor<B, 4> {
    let scores = safe_matmul(q, k.swap_dims(2, 3)) * scale;
    safe_matmul(softmax(scores, 3), v)
}
