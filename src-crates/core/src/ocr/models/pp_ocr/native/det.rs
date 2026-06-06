//! Native PP-OCRv6 text-detection model (LCNetV4 backbone + RepLKFPN neck +
//! DB head), loaded directly from HuggingFace safetensors. Produces the DB
//! probability map `[1, 1, H, W]` consumed by `postprocess_detector`.

use anyhow::Result;
use burn::nn::PaddingConfig2d;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use safetensors::SafeTensors;

use crate::ml::burn_nn::{Activation, ConvGeom, ConvLayer, ConvTransposeLayer};
use crate::ocr::models::pp_ocr::native::lcnet::{
    BlockSpec, LcnetBlock, SqueezeExcite,
};

const BACKBONE_PREFIX: &str = "model.backbone.encoder";
const NECK_GATE: Activation = Activation::HardSigmoid {
    alpha: 0.200_000_002_980_232_24,
    beta: 0.5,
};

fn upsample_nearest<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 4>,
    factor: usize,
) -> Tensor<B, 4> {
    let [_b, _c, h, w] = x.dims();
    interpolate(
        x,
        [h * factor, w * factor],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

/// Neck lateral: 1x1 in-conv then SE-residual `y + y * hard-sigmoid(gate)`.
#[derive(Debug)]
struct InsertConv<B: Backend> {
    in_conv: ConvLayer<B>,
    se: SqueezeExcite<B>,
}

impl<B: Backend<FloatElem = f32>> InsertConv<B> {
    fn load(
        tensors: &SafeTensors<'_>,
        index: usize,
        in_ch: usize,
        out_ch: usize,
        reduced: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = format!("model.neck.insert_conv.{index}");
        Ok(Self {
            in_conv: ConvLayer::conv_weight_only(
                tensors,
                &format!("{prefix}.in_conv"),
                in_ch,
                out_ch,
                ConvGeom::pointwise(),
                Activation::Identity,
                device,
            )?,
            se: SqueezeExcite::load(
                tensors,
                &format!("{prefix}.squeeze_excitation_block.conv1"),
                &format!("{prefix}.squeeze_excitation_block.conv2"),
                out_ch,
                reduced,
                NECK_GATE,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = self.in_conv.forward(x);
        let gate = self.se.gate(y.clone());
        y.clone() + y * gate
    }
}

/// Neck RepLK input-conv: 7x7 depthwise + 1x1 pointwise then SE-residual.
#[derive(Debug)]
struct InputConv<B: Backend> {
    depthwise: ConvLayer<B>,
    pointwise: ConvLayer<B>,
    se: SqueezeExcite<B>,
}

impl<B: Backend<FloatElem = f32>> InputConv<B> {
    fn load(
        tensors: &SafeTensors<'_>,
        index: usize,
        channels: usize,
        out_ch: usize,
        reduced: usize,
        dw_kernel: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = format!("model.neck.input_conv.{index}");
        Ok(Self {
            depthwise: ConvLayer::conv_bias(
                tensors,
                &format!("{prefix}.depthwise_convolution"),
                channels,
                channels,
                ConvGeom::k(dw_kernel, 1, (dw_kernel - 1) / 2, channels),
                Activation::Identity,
                device,
            )?,
            pointwise: ConvLayer::conv_weight_only(
                tensors,
                &format!("{prefix}.pointwise_convolution"),
                channels,
                out_ch,
                ConvGeom::pointwise(),
                Activation::Identity,
                device,
            )?,
            se: SqueezeExcite::load(
                tensors,
                &format!("{prefix}.squeeze_excitation_module.conv1"),
                &format!("{prefix}.squeeze_excitation_module.conv2"),
                out_ch,
                reduced,
                NECK_GATE,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let p = self.pointwise.forward(self.depthwise.forward(x));
        let gate = self.se.gate(p.clone());
        p.clone() + p * gate
    }
}

/// Native PP-OCRv6 detector.
#[derive(Debug)]
pub(crate) struct PpOcrDetector<B: Backend> {
    stem1: ConvLayer<B>,
    stem2a: ConvLayer<B>,
    stem2b: ConvLayer<B>,
    stem3: ConvLayer<B>,
    stem4: ConvLayer<B>,
    stem_pool: MaxPool2d,
    blocks: Vec<LcnetBlock<B>>,
    stage_block_counts: [usize; 4],
    insert_convs: Vec<InsertConv<B>>,
    input_convs: Vec<InputConv<B>>,
    conv_down: ConvLayer<B>,
    conv_up: ConvTransposeLayer<B>,
    conv_final: ConvTransposeLayer<B>,
}

/// Tier-specific channel widths for the small/tiny RepLKFPN detector.
pub(crate) struct DetConfig {
    /// `stem1` output width; the stem produces `2 * stem_base` channels.
    stem_base: usize,
    block_specs: fn() -> Vec<BlockSpec>,
    /// Backbone stage output channels `[C2, C3, C4, C5]`.
    stage_channels: [usize; 4],
    /// Common neck channel width (lateral / input-conv outputs share this).
    neck_width: usize,
    insert_se_reduced: usize,
    /// `input_conv` pointwise output (one quarter of the aggregate).
    input_proj: usize,
    input_se_reduced: usize,
    input_dw_kernel: usize,
    /// DB head intermediate width (`conv_down` out / `conv_up` / `conv_final` in).
    head_width: usize,
}

impl<B: Backend<FloatElem = f32>> PpOcrDetector<B> {
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        config: &DetConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let stem = |name: &str, in_ch, out_ch, geom| {
            ConvLayer::conv_bn(
                tensors,
                &format!("model.backbone.encoder.convolution.{name}"),
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

        // Neck: per backbone level (C5,C4,C3,C2) -> indices 3,2,1,0.
        let [c2, c3, c4, c5] = config.stage_channels;
        let mut insert_convs = Vec::with_capacity(4);
        for (index, in_ch) in [(3usize, c5), (2, c4), (1, c3), (0, c2)] {
            insert_convs.push(InsertConv::load(
                tensors,
                index,
                in_ch,
                config.neck_width,
                config.insert_se_reduced,
                device,
            )?);
        }
        let mut input_convs = Vec::with_capacity(4);
        for index in [3usize, 2, 1, 0] {
            input_convs.push(InputConv::load(
                tensors,
                index,
                config.neck_width,
                config.input_proj,
                config.input_se_reduced,
                config.input_dw_kernel,
                device,
            )?);
        }

        let s = config.stem_base;
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
            stage_block_counts: [2, 3, 5, 3],
            insert_convs,
            input_convs,
            conv_down: ConvLayer::conv_bn(
                tensors,
                "head.conv_down",
                "norm",
                config.neck_width,
                config.head_width,
                ConvGeom::k(3, 1, 1, 1),
                Activation::Relu,
                device,
            )?,
            conv_up: ConvTransposeLayer::load(
                tensors,
                "head.conv_up.convolution",
                "head.conv_up.convolution.bias",
                Some("head.conv_up.norm"),
                config.head_width,
                config.head_width,
                2,
                2,
                Activation::Relu,
                device,
            )?,
            conv_final: ConvTransposeLayer::load(
                tensors,
                "head.conv_final",
                "head.conv_final.bias",
                None,
                config.head_width,
                1,
                2,
                2,
                Activation::Sigmoid,
                device,
            )?,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Stem.
        let relu1 = self.stem1.forward(x);
        let relu3 = self.stem2b.forward(self.stem2a.forward(relu1.clone()));
        let pooled = self.stem_pool.forward(relu1);
        let fused = Tensor::cat(vec![pooled, relu3], 1);
        let mut feature = self.stem4.forward(self.stem3.forward(fused));

        // Backbone stages — capture each stage output (C2,C3,C4,C5).
        let mut stage_outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(4);
        let mut block_index = 0;
        for count in self.stage_block_counts {
            for _ in 0..count {
                feature = self.blocks[block_index].forward(feature);
                block_index += 1;
            }
            stage_outputs.push(feature.clone());
        }
        // stage_outputs = [C2(/4), C3(/8), C4(/16), C5(/32)].
        let c2 = stage_outputs[0].clone();
        let c3 = stage_outputs[1].clone();
        let c4 = stage_outputs[2].clone();
        let c5 = stage_outputs[3].clone();

        // Neck laterals (insert_convs are ordered for levels 3,2,1,0).
        let lat5 = self.insert_convs[0].forward(c5);
        let lat4 = self.insert_convs[1].forward(c4);
        let lat3 = self.insert_convs[2].forward(c3);
        let lat2 = self.insert_convs[3].forward(c2);

        // Top-down FPN (nearest x2 + add).
        let p5 = lat5;
        let p4 = lat4 + upsample_nearest(p5.clone(), 2);
        let p3 = lat3 + upsample_nearest(p4.clone(), 2);
        let p2 = lat2 + upsample_nearest(p3.clone(), 2);

        // RepLK input convs (ordered for levels 3,2,1,0).
        let out5 = self.input_convs[0].forward(p5);
        let out4 = self.input_convs[1].forward(p4);
        let out3 = self.input_convs[2].forward(p3);
        let out2 = self.input_convs[3].forward(p2);

        // Aggregate to /4 and concat.
        let aggregated = Tensor::cat(
            vec![
                upsample_nearest(out5, 8),
                upsample_nearest(out4, 4),
                upsample_nearest(out3, 2),
                out2,
            ],
            1,
        );

        // DB head.
        let down = self.conv_down.forward(aggregated);
        let up = self.conv_up.forward(down);
        self.conv_final.forward(up)
    }
}

/// Returns the small/tiny RepLKFPN detector config. Medium uses a different
/// neck architecture (LKPAN — see `det_medium`) and is handled separately.
pub(crate) fn det_config(detector: crate::ocr::OcrDetector) -> DetConfig {
    use crate::ocr::OcrDetector;
    match detector {
        OcrDetector::PpOcrV6SmallDet => DetConfig {
            stem_base: 24,
            block_specs: small_block_specs,
            stage_channels: [48, 96, 192, 384],
            neck_width: 96,
            insert_se_reduced: 24,
            input_proj: 24,
            input_se_reduced: 6,
            input_dw_kernel: 7,
            head_width: 24,
        },
        OcrDetector::PpOcrV6TinyDet => DetConfig {
            stem_base: 16,
            block_specs: tiny_block_specs,
            stage_channels: [32, 48, 64, 160],
            neck_width: 64,
            insert_se_reduced: 16,
            input_proj: 16,
            input_se_reduced: 4,
            input_dw_kernel: 5,
            head_width: 16,
        },
        OcrDetector::PpOcrV6MediumDet => unreachable!(
            "medium detector uses the LKPAN variant, not DetConfig"
        ),
    }
}

/// Backbone block table for the tiny tier.
fn tiny_block_specs() -> Vec<BlockSpec> {
    let b = |stage,
             idx,
             in_ch,
             token_ch,
             hidden,
             out_ch,
             downsample: bool,
             se_reduced| BlockSpec {
        stage,
        idx,
        in_ch,
        token_ch,
        hidden,
        out_ch,
        downsample_stride: downsample.then_some([2, 2]),
        se_reduced,
    };
    vec![
        b(0, 0, 32, 32, 64, 32, false, Some(8)),
        b(0, 1, 32, 32, 64, 32, false, None),
        b(1, 0, 32, 32, 64, 48, true, None),
        b(1, 1, 48, 48, 96, 48, false, Some(12)),
        b(1, 2, 48, 48, 96, 48, false, None),
        b(2, 0, 48, 48, 96, 64, true, None),
        b(2, 1, 64, 64, 128, 64, false, Some(16)),
        b(2, 2, 64, 64, 128, 64, false, None),
        b(2, 3, 64, 64, 128, 64, false, Some(16)),
        b(2, 4, 64, 64, 128, 64, false, None),
        b(3, 0, 64, 64, 128, 160, true, None),
        b(3, 1, 160, 160, 320, 160, false, Some(40)),
        b(3, 2, 160, 160, 320, 160, false, None),
    ]
}

/// Backbone block table for the small tier (downsample stride `[2, 2]`).
fn small_block_specs() -> Vec<BlockSpec> {
    let b = |stage,
             idx,
             in_ch,
             token_ch,
             hidden,
             out_ch,
             downsample: bool,
             se_reduced| BlockSpec {
        stage,
        idx,
        in_ch,
        token_ch,
        hidden,
        out_ch,
        downsample_stride: downsample.then_some([2, 2]),
        se_reduced,
    };
    vec![
        // stage 0 (/4, 48ch)
        b(0, 0, 48, 48, 96, 48, false, Some(12)),
        b(0, 1, 48, 48, 96, 48, false, None),
        // stage 1 (/8, 96ch)
        b(1, 0, 48, 48, 96, 96, true, None),
        b(1, 1, 96, 96, 192, 96, false, Some(24)),
        b(1, 2, 96, 96, 192, 96, false, None),
        // stage 2 (/16, 192ch)
        b(2, 0, 96, 96, 192, 192, true, None),
        b(2, 1, 192, 192, 384, 192, false, Some(48)),
        b(2, 2, 192, 192, 384, 192, false, None),
        b(2, 3, 192, 192, 384, 192, false, Some(48)),
        b(2, 4, 192, 192, 384, 192, false, None),
        // stage 3 (/32, 384ch)
        b(3, 0, 192, 192, 384, 384, true, None),
        b(3, 1, 384, 384, 768, 384, false, Some(96)),
        b(3, 2, 384, 384, 768, 384, false, None),
    ]
}
