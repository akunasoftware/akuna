//! Shared backbone building blocks for the PP-OCRv6 detector and recognizer.

use anyhow::Result;
use burn::nn::PaddingConfig2d;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use safetensors::SafeTensors;

use crate::ml::burn_nn::{Activation, ConvGeom, ConvLayer};

/// Backbone squeeze-excitation gate activation.
pub(super) const BACKBONE_GATE: Activation = Activation::HardSigmoid {
    alpha: 0.166_666_701_436_042_79,
    beta: 0.5,
};

/// Squeeze-excitation channel gate.
#[derive(Debug)]
pub(super) struct SqueezeExcite<B: Backend> {
    reduce: ConvLayer<B>,
    expand: ConvLayer<B>,
}

impl<B: Backend<FloatElem = f32>> SqueezeExcite<B> {
    pub(super) fn load(
        tensors: &SafeTensors<'_>,
        reduce_prefix: &str,
        expand_prefix: &str,
        channels: usize,
        reduced: usize,
        gate: Activation,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            reduce: ConvLayer::conv_bias(
                tensors,
                reduce_prefix,
                channels,
                reduced,
                ConvGeom::pointwise(),
                Activation::Relu,
                device,
            )?,
            expand: ConvLayer::conv_bias(
                tensors,
                expand_prefix,
                reduced,
                channels,
                ConvGeom::pointwise(),
                gate,
                device,
            )?,
        })
    }

    pub(super) fn gate(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let pooled = x.mean_dim(3).mean_dim(2);
        self.expand.forward(self.reduce.forward(pooled))
    }
}

/// Static description of one backbone block.
pub(super) struct BlockSpec {
    pub stage: usize,
    pub idx: usize,
    pub in_ch: usize,
    pub token_ch: usize,
    pub hidden: usize,
    pub out_ch: usize,
    /// `Some(stride)` marks a downsample block; `None` for normal blocks.
    pub downsample_stride: Option<[usize; 2]>,
    pub se_reduced: Option<usize>,
}

/// One backbone block.
#[derive(Debug)]
pub(super) struct LcnetBlock<B: Backend> {
    token_conv: ConvLayer<B>,
    se: Option<SqueezeExcite<B>>,
    channel_conv1: ConvLayer<B>,
    channel_conv2: ConvLayer<B>,
    residual: bool,
}

impl<B: Backend<FloatElem = f32>> LcnetBlock<B> {
    pub(super) fn load(
        tensors: &SafeTensors<'_>,
        prefix_root: &str,
        spec: &BlockSpec,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix =
            format!("{prefix_root}.blocks.{}.blocks.{}", spec.stage, spec.idx);
        let token_conv = match spec.downsample_stride {
            Some(stride) => ConvLayer::conv_bn(
                tensors,
                &format!("{prefix}.token_conv"),
                "normalization",
                spec.in_ch,
                spec.in_ch,
                ConvGeom {
                    kernel: [3, 3],
                    stride,
                    padding: PaddingConfig2d::Explicit(1, 1, 1, 1),
                    groups: spec.in_ch,
                },
                Activation::Identity,
                device,
            )?,
            None => ConvLayer::conv_bias(
                tensors,
                &format!("{prefix}.token_conv"),
                spec.token_ch,
                spec.token_ch,
                ConvGeom::k(3, 1, 1, spec.token_ch),
                Activation::Identity,
                device,
            )?,
        };
        let se = match spec.se_reduced {
            Some(reduced) => Some(SqueezeExcite::load(
                tensors,
                &format!("{prefix}.token_squeeze_excitation.convolutions.0"),
                &format!("{prefix}.token_squeeze_excitation.convolutions.2"),
                spec.token_ch,
                reduced,
                BACKBONE_GATE,
                device,
            )?),
            None => None,
        };
        let channel_in = if spec.downsample_stride.is_some() {
            spec.in_ch
        } else {
            spec.token_ch
        };
        Ok(Self {
            token_conv,
            se,
            channel_conv1: ConvLayer::conv_bn(
                tensors,
                &format!("{prefix}.channel_conv1"),
                "normalization",
                channel_in,
                spec.hidden,
                ConvGeom::pointwise(),
                Activation::Gelu,
                device,
            )?,
            channel_conv2: ConvLayer::conv_bn(
                tensors,
                &format!("{prefix}.channel_conv2"),
                "normalization",
                spec.hidden,
                spec.out_ch,
                ConvGeom::pointwise(),
                Activation::Identity,
                device,
            )?,
            residual: spec.downsample_stride.is_none(),
        })
    }

    pub(super) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let token = self.token_conv.forward(x);
        let token = match &self.se {
            Some(se) => token.clone() * se.gate(token),
            None => token,
        };
        let mlp = self
            .channel_conv2
            .forward(self.channel_conv1.forward(token.clone()));
        if self.residual { token + mlp } else { mlp }
    }
}
