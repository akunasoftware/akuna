//! PP-OCRv6 text recognizer for the tiny tier.

use anyhow::Result;
use burn::nn::PaddingConfig2d;
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use safetensors::SafeTensors;

use crate::ml::burn_nn::{
    Activation, BatchNorm2dLayer, ConvGeom, ConvLayer, LinearLayer,
    apply_activation,
};
use crate::ocr::models::pp_ocr::native::lcnet::{BlockSpec, LcnetBlock};

const BACKBONE_PREFIX: &str = "model.backbone.encoder";

/// Native PP-OCRv6 tiny recognizer.
#[derive(Debug)]
pub(crate) struct PpOcrRecognizerTiny<B: Backend> {
    stem_conv1: ConvLayer<B>,
    stem_conv2: ConvLayer<B>,
    blocks: Vec<LcnetBlock<B>>,
    height_pool: AvgPool2d,
    head_conv1: ConvLayer<B>,
    head_norm1: BatchNorm2dLayer<B>,
    head_conv2: ConvLayer<B>,
    head_norm2: BatchNorm2dLayer<B>,
    fc1: LinearLayer<B>,
    fc2: LinearLayer<B>,
}

impl<B: Backend<FloatElem = f32>> PpOcrRecognizerTiny<B> {
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        num_classes: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let specs = tiny_block_specs();
        let mut blocks = Vec::with_capacity(specs.len());
        for spec in &specs {
            blocks.push(LcnetBlock::load(
                tensors,
                BACKBONE_PREFIX,
                spec,
                device,
            )?);
        }

        Ok(Self {
            stem_conv1: ConvLayer::conv_bn(
                tensors,
                &format!("{BACKBONE_PREFIX}.convolution.conv1"),
                "normalization",
                3,
                24,
                ConvGeom::k(3, 2, 1, 1),
                Activation::Gelu,
                device,
            )?,
            stem_conv2: ConvLayer::conv_bn(
                tensors,
                &format!("{BACKBONE_PREFIX}.convolution.conv2"),
                "normalization",
                24,
                48,
                ConvGeom::k(3, 2, 1, 1),
                Activation::Identity,
                device,
            )?,
            blocks,
            height_pool: AvgPool2dConfig::new([3, 2])
                .with_strides([3, 2])
                .with_padding(PaddingConfig2d::Valid)
                .with_count_include_pad(false)
                .init(),
            head_conv1: ConvLayer::conv1d_weight_only(
                tensors,
                "head.conv1",
                160,
                160,
                5,
                160,
                device,
            )?,
            head_norm1: BatchNorm2dLayer::load(
                tensors,
                "head.norm1",
                160,
                device,
            )?,
            head_conv2: ConvLayer::conv1d_weight_only(
                tensors,
                "head.conv2",
                160,
                160,
                1,
                1,
                device,
            )?,
            head_norm2: BatchNorm2dLayer::load(
                tensors,
                "head.norm2",
                160,
                device,
            )?,
            fc1: LinearLayer::load(tensors, "head.fc1", 160, 80, device)?,
            fc2: LinearLayer::load(
                tensors,
                "head.fc2",
                80,
                num_classes,
                device,
            )?,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // 2-conv stem (conv1 GELU, conv2 identity).
        let feature = self.stem_conv2.forward(self.stem_conv1.forward(x));

        // Backbone.
        let mut feature = feature;
        for block in &self.blocks {
            feature = block.forward(feature);
        }

        // Collapse height + halve width -> [B, 160, 1, W].
        let pooled = self.height_pool.forward(feature);

        // Conv head: dw5 -> BN -> hardswish; pw1 -> BN -> hardswish.
        let h = apply_activation(
            self.head_norm1.forward(self.head_conv1.forward(pooled)),
            Activation::HardSwish,
        );
        let h = apply_activation(
            self.head_norm2.forward(self.head_conv2.forward(h)),
            Activation::HardSwish,
        );

        // [B, 160, 1, W] -> [B, W, 160] -> CTC head -> softmax.
        let [batch, channels, _h, width] = h.dims();
        let sequence = h.reshape([batch, channels, width]).swap_dims(1, 2);
        let logits = self.fc2.forward(self.fc1.forward(sequence));
        softmax(logits, 2)
    }
}

/// Backbone block table for the tiny recognizer (stages `[1, 1, 3, 4]`).
fn tiny_block_specs() -> Vec<BlockSpec> {
    let normal = |stage, idx, ch, hidden, se_reduced| BlockSpec {
        stage,
        idx,
        in_ch: ch,
        token_ch: ch,
        hidden,
        out_ch: ch,
        downsample_stride: None,
        se_reduced,
    };
    let down = |stage, idx, in_ch, hidden, out_ch| BlockSpec {
        stage,
        idx,
        in_ch,
        token_ch: in_ch,
        hidden,
        out_ch,
        downsample_stride: Some([2, 1]),
        se_reduced: None,
    };
    vec![
        // stage 0 (48ch)
        normal(0, 0, 48, 96, Some(12)),
        // stage 1 (48ch)
        normal(1, 0, 48, 96, None),
        // stage 2 (48 -> 96ch)
        down(2, 0, 48, 96, 96),
        normal(2, 1, 96, 192, Some(24)),
        normal(2, 2, 96, 192, None),
        // stage 3 (96 -> 160ch)
        down(3, 0, 96, 192, 160),
        normal(3, 1, 160, 320, Some(40)),
        normal(3, 2, 160, 320, None),
        normal(3, 3, 160, 320, None),
    ]
}
