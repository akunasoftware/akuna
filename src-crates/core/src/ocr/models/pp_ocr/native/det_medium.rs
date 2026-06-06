//! Native PP-OCRv6 **medium** text detector. The backbone (LCNetV4) and DB head
//! match the small/tiny detector, but the neck is an LKPAN with large-kernel
//! projections and per-level "intraclass" multi-ratio strip-conv blocks (no SE).

use anyhow::Result;
use burn::nn::PaddingConfig2d;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};
use safetensors::SafeTensors;

use crate::ml::burn_nn::{Activation, ConvGeom, ConvLayer, ConvTransposeLayer};
use crate::ocr::models::pp_ocr::native::lcnet::{BlockSpec, LcnetBlock};

const BACKBONE_PREFIX: &str = "model.backbone.encoder";
const STEM_BASE: usize = 64;
const NECK_CH: usize = 256;
const PROJ_CH: usize = 64;
const REDUCE_CH: usize = 32;

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

/// A K×K symmetric + K×1 vertical + 1×K horizontal triple summed together.
#[derive(Debug)]
struct StripStage<B: Backend> {
    symmetric: ConvLayer<B>,
    vertical: ConvLayer<B>,
    horizontal: ConvLayer<B>,
}

impl<B: Backend<FloatElem = f32>> StripStage<B> {
    fn load(
        tensors: &SafeTensors<'_>,
        block_prefix: &str,
        ratio: &str,
        kernel: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let pad = kernel / 2;
        let conv = |name: &str, geom| {
            ConvLayer::conv_bias(
                tensors,
                &format!("{block_prefix}.{name}_{ratio}ratio"),
                REDUCE_CH,
                REDUCE_CH,
                geom,
                Activation::Identity,
                device,
            )
        };
        Ok(Self {
            symmetric: conv(
                "symmetric_conv_long",
                ConvGeom::k(kernel, 1, pad, 1),
            )?,
            vertical: conv(
                "vertical_long_to_small_conv",
                ConvGeom {
                    kernel: [kernel, 1],
                    stride: [1, 1],
                    padding: PaddingConfig2d::Explicit(pad, 0, pad, 0),
                    groups: 1,
                },
            )?,
            horizontal: conv(
                "horizontal_small_to_long_conv",
                ConvGeom {
                    kernel: [1, kernel],
                    stride: [1, 1],
                    padding: PaddingConfig2d::Explicit(0, pad, 0, pad),
                    groups: 1,
                },
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.symmetric.forward(x.clone())
            + self.vertical.forward(x.clone())
            + self.horizontal.forward(x)
    }
}

/// Per-level intraclass block: reduce → long→mid→short strip cascade →
/// conv_final(+BN+ReLU) → residual with the 64-ch block input.
#[derive(Debug)]
struct IntraclassBlock<B: Backend> {
    reduce: ConvLayer<B>,
    long: StripStage<B>,
    mid: StripStage<B>,
    short: StripStage<B>,
    conv_final: ConvLayer<B>,
}

impl<B: Backend<FloatElem = f32>> IntraclassBlock<B> {
    fn load(
        tensors: &SafeTensors<'_>,
        index: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = format!("model.neck.intraclass_blocks.{index}");
        Ok(Self {
            reduce: ConvLayer::conv_bias(
                tensors,
                &format!("{prefix}.conv_reduce_channel"),
                PROJ_CH,
                REDUCE_CH,
                ConvGeom::pointwise(),
                Activation::Identity,
                device,
            )?,
            long: StripStage::load(tensors, &prefix, "long", 7, device)?,
            mid: StripStage::load(tensors, &prefix, "mid", 5, device)?,
            short: StripStage::load(tensors, &prefix, "short", 3, device)?,
            conv_final: ConvLayer::conv_bn(
                tensors,
                &format!("{prefix}.conv_final"),
                "norm",
                REDUCE_CH,
                PROJ_CH,
                ConvGeom::pointwise(),
                Activation::Relu,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let reduced = self.reduce.forward(x.clone());
        let s1 = self.long.forward(reduced);
        let s2 = self.mid.forward(s1);
        let s3 = self.short.forward(s2);
        x + self.conv_final.forward(s3)
    }
}

/// Native PP-OCRv6 medium detector.
#[derive(Debug)]
pub(crate) struct PpOcrDetectorMedium<B: Backend> {
    stem1: ConvLayer<B>,
    stem2a: ConvLayer<B>,
    stem2b: ConvLayer<B>,
    stem3: ConvLayer<B>,
    stem4: ConvLayer<B>,
    stem_pool: MaxPool2d,
    blocks: Vec<LcnetBlock<B>>,
    stage_block_counts: [usize; 4],
    channel_adjust: Vec<ConvLayer<B>>,
    projection: Vec<ConvLayer<B>>,
    pan_head: Vec<ConvLayer<B>>,
    pan_lateral: Vec<ConvLayer<B>>,
    intraclass: Vec<IntraclassBlock<B>>,
    conv_down: ConvLayer<B>,
    conv_up: ConvTransposeLayer<B>,
    conv_final: ConvTransposeLayer<B>,
}

impl<B: Backend<FloatElem = f32>> PpOcrDetectorMedium<B> {
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
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

        let specs = medium_block_specs();
        let mut blocks = Vec::with_capacity(specs.len());
        for spec in &specs {
            blocks.push(LcnetBlock::load(
                tensors,
                BACKBONE_PREFIX,
                spec,
                device,
            )?);
        }

        // Neck. Index 0 = shallowest level (C2), matching channel widths.
        let stage_channels = [128usize, 256, 512, 896];
        let mut channel_adjust = Vec::with_capacity(4);
        for (index, &in_ch) in stage_channels.iter().enumerate() {
            channel_adjust.push(ConvLayer::conv_weight_only(
                tensors,
                &format!(
                    "model.neck.input_channel_adjustment_convolution.{index}"
                ),
                in_ch,
                NECK_CH,
                ConvGeom::pointwise(),
                Activation::Identity,
                device,
            )?);
        }
        let mut projection = Vec::with_capacity(4);
        let mut pan_lateral = Vec::with_capacity(4);
        for index in 0..4 {
            projection.push(ConvLayer::conv_bias(
                tensors,
                &format!(
                    "model.neck.input_feature_projection_convolution.{index}"
                ),
                NECK_CH,
                PROJ_CH,
                ConvGeom::k(9, 1, 4, 1),
                Activation::Identity,
                device,
            )?);
            pan_lateral.push(ConvLayer::conv_bias(
                tensors,
                &format!(
                    "model.neck.path_aggregation_lateral_convolution.{index}"
                ),
                PROJ_CH,
                PROJ_CH,
                ConvGeom::k(9, 1, 4, 1),
                Activation::Identity,
                device,
            )?);
        }
        let mut pan_head = Vec::with_capacity(3);
        for index in 0..3 {
            pan_head.push(ConvLayer::conv_weight_only(
                tensors,
                &format!(
                    "model.neck.path_aggregation_head_convolution.{index}"
                ),
                PROJ_CH,
                PROJ_CH,
                ConvGeom::k(3, 2, 1, 1),
                Activation::Identity,
                device,
            )?);
        }
        let mut intraclass = Vec::with_capacity(4);
        for index in 0..4 {
            intraclass.push(IntraclassBlock::load(tensors, index, device)?);
        }

        let s = STEM_BASE;
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
            channel_adjust,
            projection,
            pan_head,
            pan_lateral,
            intraclass,
            conv_down: ConvLayer::conv_bn(
                tensors,
                "head.conv_down",
                "norm",
                NECK_CH,
                PROJ_CH,
                ConvGeom::k(3, 1, 1, 1),
                Activation::Relu,
                device,
            )?,
            conv_up: ConvTransposeLayer::load(
                tensors,
                "head.conv_up.convolution",
                "head.conv_up.convolution.bias",
                Some("head.conv_up.norm"),
                PROJ_CH,
                PROJ_CH,
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
                PROJ_CH,
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

        // Backbone — capture stage outputs (C2,C3,C4,C5).
        let mut levels: Vec<Tensor<B, 4>> = Vec::with_capacity(4);
        let mut block_index = 0;
        for count in self.stage_block_counts {
            for _ in 0..count {
                feature = self.blocks[block_index].forward(feature);
                block_index += 1;
            }
            levels.push(feature.clone());
        }

        // Channel adjust -> 256, indexed 0=C2 .. 3=C5.
        let lat: Vec<Tensor<B, 4>> = self
            .channel_adjust
            .iter()
            .zip(levels)
            .map(|(conv, level)| conv.forward(level))
            .collect();

        // Top-down FPN (nearest x2 + add), deepest-first.
        let p5 = lat[3].clone();
        let p4 = lat[2].clone() + upsample_nearest(p5.clone(), 2);
        let p3 = lat[1].clone() + upsample_nearest(p4.clone(), 2);
        let p2 = lat[0].clone() + upsample_nearest(p3.clone(), 2);

        // Per-level 9x9 projection -> 64.
        let proj2 = self.projection[0].forward(p2);
        let proj3 = self.projection[1].forward(p3);
        let proj4 = self.projection[2].forward(p4);
        let proj5 = self.projection[3].forward(p5);

        // Bottom-up PAN: 3x3 stride-2 head adds, shallow->deep.
        let n2 = proj2;
        let n3 = proj3 + self.pan_head[0].forward(n2.clone());
        let n4 = proj4 + self.pan_head[1].forward(n3.clone());
        let n5 = proj5 + self.pan_head[2].forward(n4.clone());

        // 9x9 lateral then intraclass block per level.
        let ic2 = self.intraclass[0].forward(self.pan_lateral[0].forward(n2));
        let ic3 = self.intraclass[1].forward(self.pan_lateral[1].forward(n3));
        let ic4 = self.intraclass[2].forward(self.pan_lateral[2].forward(n4));
        let ic5 = self.intraclass[3].forward(self.pan_lateral[3].forward(n5));

        // Aggregate to stride-4, deepest-first concat.
        let aggregated = Tensor::cat(
            vec![
                upsample_nearest(ic5, 8),
                upsample_nearest(ic4, 4),
                upsample_nearest(ic3, 2),
                ic2,
            ],
            1,
        );

        // DB head.
        let down = self.conv_down.forward(aggregated);
        let up = self.conv_up.forward(down);
        self.conv_final.forward(up)
    }
}

/// Backbone block table for the medium tier (downsample stride `[2, 2]`).
fn medium_block_specs() -> Vec<BlockSpec> {
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
        b(0, 0, 128, 128, 256, 128, false, Some(32)),
        b(0, 1, 128, 128, 256, 128, false, None),
        b(1, 0, 128, 128, 256, 256, true, None),
        b(1, 1, 256, 256, 512, 256, false, Some(64)),
        b(1, 2, 256, 256, 512, 256, false, None),
        b(2, 0, 256, 256, 512, 512, true, None),
        b(2, 1, 512, 512, 1024, 512, false, Some(128)),
        b(2, 2, 512, 512, 1024, 512, false, None),
        b(2, 3, 512, 512, 1024, 512, false, Some(128)),
        b(2, 4, 512, 512, 1024, 512, false, None),
        b(3, 0, 512, 512, 1024, 896, true, None),
        b(3, 1, 896, 896, 1792, 896, false, Some(224)),
        b(3, 2, 896, 896, 1792, 896, false, None),
    ]
}
