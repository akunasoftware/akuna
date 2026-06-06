//! Native PP-OCRv6 text-recognition model (LCNetV4 backbone -> light-SVTR neck
//! -> CTC head), loaded from HuggingFace safetensors. Produces softmax CTC
//! posteriors `[1, seq, num_classes]` consumed by `postprocess_recognizer`.

use anyhow::Result;
use burn::nn::PaddingConfig2d;
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use safetensors::SafeTensors;

use crate::ml::burn_nn::{
    Activation, ConvGeom, ConvLayer, LayerNormLayer, LinearLayer,
    scaled_dot_product_attention, silu,
};
use crate::ocr::models::pp_ocr::native::lcnet::{BlockSpec, LcnetBlock};

const BACKBONE_PREFIX: &str = "model.backbone.encoder";
const HEADS: usize = 8;
const SVTR_EPS: f64 = 1e-5;
const FINAL_NORM_EPS: f64 = 1e-6;

/// Tier-specific recognizer widths (small / medium light-SVTR architecture).
pub(crate) struct RecConfig {
    /// `stem1` output width; the stem produces `2 * stem_base` channels.
    stem_base: usize,
    block_specs: fn() -> Vec<BlockSpec>,
    /// Backbone output channels (the neck conv-block input).
    backbone_out: usize,
    /// SVTR embedding dim (= neck conv-block output).
    dim: usize,
    head_dim: usize,
    /// SVTR MLP hidden width.
    mlp_hidden: usize,
}

/// One pre-norm SVTR transformer block.
#[derive(Debug)]
struct SvtrBlock<B: Backend> {
    norm1: LayerNormLayer<B>,
    qkv: LinearLayer<B>,
    projection: LinearLayer<B>,
    norm2: LayerNormLayer<B>,
    fc1: LinearLayer<B>,
    fc2: LinearLayer<B>,
    dim: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend<FloatElem = f32>> SvtrBlock<B> {
    fn load(
        tensors: &SafeTensors<'_>,
        index: usize,
        config: &RecConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = format!("head.encoder.svtr_block.{index}");
        let dim = config.dim;
        Ok(Self {
            norm1: LayerNormLayer::load(
                tensors,
                &format!("{prefix}.layer_norm1"),
                dim,
                SVTR_EPS,
                device,
            )?,
            qkv: LinearLayer::load(
                tensors,
                &format!("{prefix}.self_attn.qkv"),
                dim,
                dim * 3,
                device,
            )?,
            projection: LinearLayer::load(
                tensors,
                &format!("{prefix}.self_attn.projection"),
                dim,
                dim,
                device,
            )?,
            norm2: LayerNormLayer::load(
                tensors,
                &format!("{prefix}.layer_norm2"),
                dim,
                SVTR_EPS,
                device,
            )?,
            fc1: LinearLayer::load(
                tensors,
                &format!("{prefix}.mlp.fc1"),
                dim,
                config.mlp_hidden,
                device,
            )?,
            fc2: LinearLayer::load(
                tensors,
                &format!("{prefix}.mlp.fc2"),
                config.mlp_hidden,
                dim,
                device,
            )?,
            dim,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _dim] = x.dims();
        // Self-attention.
        let normed = self.norm1.forward(x.clone());
        let qkv = self.qkv.forward(normed).reshape([
            batch,
            seq,
            3,
            HEADS,
            self.head_dim,
        ]);
        let head = |slot: usize| {
            qkv.clone()
                .narrow(2, slot, 1)
                .reshape([batch, seq, HEADS, self.head_dim])
                .swap_dims(1, 2)
        };
        let context =
            scaled_dot_product_attention(head(0), head(1), head(2), self.scale)
                .swap_dims(1, 2)
                .reshape([batch, seq, self.dim]);
        let x = x + self.projection.forward(context);
        // MLP.
        let normed = self.norm2.forward(x.clone());
        let mlp = self.fc2.forward(silu(self.fc1.forward(normed)));
        x + mlp
    }
}

/// Native PP-OCRv6 recognizer.
#[derive(Debug)]
pub(crate) struct PpOcrRecognizer<B: Backend> {
    stem1: ConvLayer<B>,
    stem2a: ConvLayer<B>,
    stem2b: ConvLayer<B>,
    stem3: ConvLayer<B>,
    stem4: ConvLayer<B>,
    stem_pool: MaxPool2d,
    blocks: Vec<LcnetBlock<B>>,
    height_pool: AvgPool2d,
    conv_block0: ConvLayer<B>,
    conv_block1: ConvLayer<B>,
    conv_block2: ConvLayer<B>,
    svtr: Vec<SvtrBlock<B>>,
    final_norm: LayerNormLayer<B>,
    head: LinearLayer<B>,
    dim: usize,
}

impl<B: Backend<FloatElem = f32>> PpOcrRecognizer<B> {
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        config: &RecConfig,
        num_classes: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let stem = |name: &str, in_ch, out_ch, geom| {
            ConvLayer::conv_bn(
                tensors,
                &format!("{BACKBONE_PREFIX}.convolution.{name}"),
                "normalization",
                in_ch,
                out_ch,
                geom,
                Activation::Relu,
                device,
            )
        };
        let asym = ConvGeom {
            kernel: [2, 2],
            stride: [1, 1],
            padding: PaddingConfig2d::Explicit(0, 0, 1, 1),
            groups: 1,
        };

        let specs = (config.block_specs)();
        let mut blocks = Vec::with_capacity(specs.len());
        for spec in &specs {
            blocks.push(LcnetBlock::load(
                tensors,
                BACKBONE_PREFIX,
                spec,
                device,
            )?);
        }

        let mut svtr = Vec::with_capacity(2);
        for index in 0..2 {
            svtr.push(SvtrBlock::load(tensors, index, config, device)?);
        }

        let s = config.stem_base;
        let dim = config.dim;
        let neck_conv_block = |idx: usize, in_ch, out_ch, geom| {
            ConvLayer::conv_bn(
                tensors,
                &format!("head.encoder.conv_block.{idx}"),
                "normalization",
                in_ch,
                out_ch,
                geom,
                Activation::Silu,
                device,
            )
        };
        Ok(Self {
            stem1: stem("stem1", 3, s, ConvGeom::k(3, 2, 1, 1))?,
            stem2a: stem("stem2a", s, s / 2, asym.clone())?,
            stem2b: stem("stem2b", s / 2, s, asym)?,
            stem3: stem("stem3", 2 * s, s, ConvGeom::k(3, 2, 1, 1))?,
            stem4: stem("stem4", s, 2 * s, ConvGeom::pointwise())?,
            stem_pool: MaxPool2dConfig::new([2, 2])
                .with_strides([1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
                .init(),
            blocks,
            height_pool: AvgPool2dConfig::new([3, 2])
                .with_strides([3, 2])
                .with_padding(PaddingConfig2d::Valid)
                .with_count_include_pad(false)
                .init(),
            conv_block0: neck_conv_block(
                0,
                config.backbone_out,
                dim,
                ConvGeom::pointwise(),
            )?,
            conv_block1: neck_conv_block(
                1,
                config.backbone_out,
                dim,
                ConvGeom::pointwise(),
            )?,
            conv_block2: neck_conv_block(
                2,
                dim,
                dim,
                ConvGeom {
                    kernel: [1, 7],
                    stride: [1, 1],
                    padding: PaddingConfig2d::Explicit(0, 3, 0, 3),
                    groups: dim,
                },
            )?,
            svtr,
            final_norm: LayerNormLayer::load(
                tensors,
                "head.encoder.norm",
                dim,
                FINAL_NORM_EPS,
                device,
            )?,
            head: LinearLayer::load(
                tensors,
                "head.head",
                dim,
                num_classes,
                device,
            )?,
            dim,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // Stem.
        let relu1 = self.stem1.forward(x);
        let relu3 = self.stem2b.forward(self.stem2a.forward(relu1.clone()));
        let pooled = self.stem_pool.forward(relu1);
        let fused = Tensor::cat(vec![pooled, relu3], 1);
        let mut feature = self.stem4.forward(self.stem3.forward(fused));

        // Backbone.
        for block in &self.blocks {
            feature = block.forward(feature);
        }

        // Collapse height + halve width.
        let feature = self.height_pool.forward(feature);
        let [batch, _c, _h, width] = feature.dims();

        // Light-conv neck.
        let shortcut = self.conv_block0.forward(feature.clone());
        let m1 = self.conv_block1.forward(feature);
        let m2 = self.conv_block2.forward(m1.clone());
        let feat = m1 + m2;

        // Flatten to a [B, seq, dim] sequence.
        let mut sequence =
            feat.reshape([batch, self.dim, width]).swap_dims(1, 2);

        // SVTR blocks + final norm.
        for block in &self.svtr {
            sequence = block.forward(sequence);
        }
        let sequence = self.final_norm.forward(sequence);

        // Reshape back to [B, dim, 1, seq], add the conv_block.0 shortcut.
        let back = sequence
            .reshape([batch, 1, width, self.dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3)
            + shortcut;

        // [B, seq, dim] -> CTC head -> softmax over classes.
        let sequence = back.reshape([batch, self.dim, width]).swap_dims(1, 2);
        let logits = self.head.forward(sequence);
        softmax(logits, 2)
    }
}

/// Backbone block table for the small tier (downsample stride `[2, 1]`).
fn small_block_specs() -> Vec<BlockSpec> {
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
        // stage 0 (96ch)
        normal(0, 0, 96, 192, Some(24)),
        // stage 1 (96ch)
        normal(1, 0, 96, 192, None),
        normal(1, 1, 96, 192, None),
        // stage 2 (96 -> 192ch)
        down(2, 0, 96, 192, 192),
        normal(2, 1, 192, 384, Some(48)),
        normal(2, 2, 192, 384, None),
        normal(2, 3, 192, 384, Some(48)),
        normal(2, 4, 192, 384, None),
        normal(2, 5, 192, 384, Some(48)),
        normal(2, 6, 192, 384, None),
        // stage 3 (192 -> 384ch)
        down(3, 0, 192, 384, 384),
        normal(3, 1, 384, 768, Some(96)),
        normal(3, 2, 384, 768, None),
    ]
}

/// Returns the small/medium light-SVTR recognizer config. Tiny uses a different
/// head architecture (conv-only — see `rec_tiny`) and is handled separately.
pub(crate) fn rec_config(recognizer: crate::ocr::OcrRecognizer) -> RecConfig {
    use crate::ocr::OcrRecognizer;
    match recognizer {
        OcrRecognizer::PpOcrV6SmallRec => RecConfig {
            stem_base: 48,
            block_specs: small_block_specs,
            backbone_out: 384,
            dim: 120,
            head_dim: 15,
            mlp_hidden: 240,
        },
        OcrRecognizer::PpOcrV6MediumRec => RecConfig {
            stem_base: 64,
            block_specs: medium_block_specs,
            backbone_out: 768,
            dim: 192,
            head_dim: 24,
            mlp_hidden: 768,
        },
        OcrRecognizer::PpOcrV6TinyRec => unreachable!(
            "tiny recognizer uses the conv-only head variant, not RecConfig"
        ),
    }
}

/// Backbone block table for the medium tier. Stage 1 opens with a stride-`[1,1]`
/// transition block (channel change, no residual); spatial downsamples are at
/// 2.0 and 3.0.
fn medium_block_specs() -> Vec<BlockSpec> {
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
    let down = |stage, idx, in_ch, hidden, out_ch, stride| BlockSpec {
        stage,
        idx,
        in_ch,
        token_ch: in_ch,
        hidden,
        out_ch,
        downsample_stride: Some(stride),
        se_reduced: None,
    };
    vec![
        // stage 0 (128ch)
        normal(0, 0, 128, 256, Some(32)),
        // stage 1 (128 -> 256ch, transition has stride [1,1])
        down(1, 0, 128, 256, 256, [1, 1]),
        normal(1, 1, 256, 512, None),
        normal(1, 2, 256, 512, Some(64)),
        // stage 2 (256 -> 512ch)
        down(2, 0, 256, 512, 512, [2, 1]),
        normal(2, 1, 512, 1024, Some(128)),
        normal(2, 2, 512, 1024, None),
        normal(2, 3, 512, 1024, Some(128)),
        normal(2, 4, 512, 1024, None),
        normal(2, 5, 512, 1024, Some(128)),
        normal(2, 6, 512, 1024, None),
        // stage 3 (512 -> 768ch)
        down(3, 0, 512, 1024, 768, [2, 1]),
        normal(3, 1, 768, 1536, Some(192)),
        normal(3, 2, 768, 1536, None),
    ]
}
