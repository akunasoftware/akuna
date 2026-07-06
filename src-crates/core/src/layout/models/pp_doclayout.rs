#![allow(
    clippy::large_enum_variant,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::module::Param;
use burn::nn::{PaddingConfig2d, conv::Conv2d};
use burn::tensor::{
    Tensor, TensorData,
    activation::softmax,
    backend::Backend,
    module::{interpolate, max_pool2d},
    ops::{InterpolateMode, InterpolateOptions, PadMode},
};
use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};
use image::{DynamicImage, GenericImageView};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::ml::burn_nn::{
    batch_norm_inference, gelu, relu, sigmoid as sigmoid_tensor, silu,
};
use crate::ml::{safe_matmul, sigmoid_f32};

const PP_DOCLAYOUT_REPO_ID: &str = "PaddlePaddle/PP-DocLayoutV3_safetensors";
const PP_DOCLAYOUT_CONFIG: &str = "config.json";
const PP_DOCLAYOUT_PREPROCESSOR: &str = "preprocessor_config.json";
const PP_DOCLAYOUT_WEIGHTS: &str = "model.safetensors";
const PP_DOCLAYOUT_WEIGHTS_ENV: &str = "PP_DOCLAYOUT_WEIGHTS";

#[derive(Debug, Clone)]
struct PpDocLayoutFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) preprocessor_path: PathBuf,
    pub(crate) weights_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct PpDocLayoutConfig {
    pub(crate) model_type: String,
    pub(crate) architectures: Vec<String>,
    pub(crate) d_model: usize,
    pub(crate) num_queries: usize,
    pub(crate) decoder_layers: usize,
    pub(crate) decoder_attention_heads: usize,
    pub(crate) decoder_n_points: usize,
    pub(crate) feature_strides: Vec<usize>,
    pub(crate) id2label: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpDocLayoutPreprocessorConfig {
    pub(crate) do_resize: bool,
    pub(crate) size: PpDocLayoutSize,
    pub(crate) image_mean: Vec<f32>,
    pub(crate) image_std: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PpDocLayoutSize {
    pub(crate) height: usize,
    pub(crate) width: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LayoutDetection {
    pub(crate) label: String,
    pub(crate) score: f32,
    pub(crate) bbox: [f32; 4],
    pub(crate) order: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayoutInput {
    pub(crate) values: Vec<f32>,
    pub(crate) channels: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) original_height: u32,
    pub(crate) original_width: u32,
}

#[derive(Debug)]
pub(crate) struct PpConv1x1BatchNorm<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_bias: Tensor<B, 1>,
    running_mean: Tensor<B, 1>,
    running_var: Tensor<B, 1>,
    epsilon: f64,
}

#[derive(Debug)]
pub(crate) struct PpEncoderInputProjection<B: Backend> {
    projections: Vec<PpConv1x1BatchNorm<B>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PpActivation {
    Identity,
    Relu,
    Silu,
}

#[derive(Debug)]
pub(crate) struct PpConvBatchNorm<B: Backend> {
    conv: Conv2d<B>,
    norm_weight: Tensor<B, 1>,
    norm_bias: Tensor<B, 1>,
    running_mean: Tensor<B, 1>,
    running_var: Tensor<B, 1>,
    epsilon: f64,
    activation: PpActivation,
}

#[derive(Debug)]
pub(crate) struct PpHgnetStem<B: Backend> {
    stem1: PpConvBatchNorm<B>,
    stem2a: PpConvBatchNorm<B>,
    stem2b: PpConvBatchNorm<B>,
    stem3: PpConvBatchNorm<B>,
    stem4: PpConvBatchNorm<B>,
}

#[derive(Debug)]
pub(crate) struct PpHgnetBasicLayer<B: Backend> {
    layers: Vec<PpHgnetLayer<B>>,
    aggregation_squeeze: PpConvBatchNorm<B>,
    aggregation_excitation: PpConvBatchNorm<B>,
    residual: bool,
}

#[derive(Debug)]
pub(crate) enum PpHgnetLayer<B: Backend> {
    Conv(PpConvBatchNorm<B>),
    Light {
        pointwise: PpConvBatchNorm<B>,
        depthwise: PpConvBatchNorm<B>,
    },
}

#[derive(Debug)]
pub(crate) struct PpHgnetStage<B: Backend> {
    downsample: Option<PpConvBatchNorm<B>>,
    blocks: Vec<PpHgnetBasicLayer<B>>,
}

#[derive(Debug)]
pub(crate) struct PpHgnetBackbone<B: Backend> {
    stem: PpHgnetStem<B>,
    stage0: PpHgnetStage<B>,
    stage1: PpHgnetStage<B>,
    stage2: PpHgnetStage<B>,
    stage3: PpHgnetStage<B>,
}

#[derive(Debug)]
pub(crate) struct PpBackboneFeatureProjector<B: Backend> {
    backbone: PpHgnetBackbone<B>,
    projection: PpEncoderInputProjection<B>,
}

#[derive(Debug)]
pub(crate) struct PpHybridEncoderConvs<B: Backend> {
    aifi: PpAifiLayer<B>,
    lateral_convs: Vec<PpConvBatchNorm<B>>,
    downsample_convs: Vec<PpConvBatchNorm<B>>,
    fpn_blocks: Vec<PpCspRepLayer<B>>,
    pan_blocks: Vec<PpCspRepLayer<B>>,
}

#[derive(Debug)]
pub(crate) struct PpLayerNorm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
    epsilon: f64,
}

#[derive(Debug)]
pub(crate) struct PpAifiLayer<B: Backend> {
    q_proj: PpLinear<B>,
    k_proj: PpLinear<B>,
    v_proj: PpLinear<B>,
    out_proj: PpLinear<B>,
    self_attn_layer_norm: PpLayerNorm<B>,
    fc1: PpLinear<B>,
    fc2: PpLinear<B>,
    final_layer_norm: PpLayerNorm<B>,
}

#[derive(Debug)]
pub(crate) struct PpRepVggBlock<B: Backend> {
    conv1: PpConvBatchNorm<B>,
    conv2: PpConvBatchNorm<B>,
}

#[derive(Debug)]
pub(crate) struct PpCspRepLayer<B: Backend> {
    conv1: PpConvBatchNorm<B>,
    conv2: PpConvBatchNorm<B>,
    bottlenecks: Vec<PpRepVggBlock<B>>,
}

#[derive(Debug)]
pub(crate) struct PpLinear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

#[derive(Debug)]
pub(crate) struct PpMlp<B: Backend> {
    layers: Vec<PpLinear<B>>,
}

#[derive(Debug)]
pub(crate) struct PpEncoderDetectionHead<B: Backend> {
    decoder_input_proj: PpEncoderInputProjection<B>,
    enc_output: PpLinear<B>,
    enc_output_norm: PpLayerNorm<B>,
    enc_score_head: PpLinear<B>,
    enc_bbox_head: PpMlp<B>,
}

#[derive(Debug)]
pub(crate) struct PpDecoderAttention<B: Backend> {
    q_proj: PpLinear<B>,
    k_proj: PpLinear<B>,
    v_proj: PpLinear<B>,
    out_proj: PpLinear<B>,
}

#[derive(Debug)]
pub(crate) struct PpDecoderCrossAttention<B: Backend> {
    sampling_offsets: PpLinear<B>,
    attention_weights: PpLinear<B>,
    value_proj: PpLinear<B>,
    output_proj: PpLinear<B>,
}

#[derive(Debug)]
pub(crate) struct PpDecoderLayer<B: Backend> {
    self_attn: PpDecoderAttention<B>,
    self_attn_layer_norm: PpLayerNorm<B>,
    encoder_attn: PpDecoderCrossAttention<B>,
    encoder_attn_layer_norm: PpLayerNorm<B>,
    fc1: PpLinear<B>,
    fc2: PpLinear<B>,
    final_layer_norm: PpLayerNorm<B>,
}

#[derive(Debug)]
pub(crate) struct PpDocLayoutDecoder<B: Backend> {
    query_pos_head: PpMlp<B>,
    layers: Vec<PpDecoderLayer<B>>,
    order_head: Vec<PpLinear<B>>,
    decoder_norm: PpLayerNorm<B>,
    global_pointer: PpLinear<B>,
}

#[derive(Debug)]
pub(crate) struct PpDocLayoutDetector<B: Backend> {
    backbone: PpBackboneFeatureProjector<B>,
    encoder: PpHybridEncoderConvs<B>,
    detection_head: PpEncoderDetectionHead<B>,
    decoder: PpDocLayoutDecoder<B>,
}

#[derive(Debug)]
pub(crate) struct PpDocLayoutRawOutput<B: Backend> {
    scores: Tensor<B, 3>,
    boxes: Tensor<B, 3>,
    order_features: Tensor<B, 3>,
}

#[derive(Debug)]
pub(crate) struct PpDocLayoutRuntime<B: Backend> {
    config: PpDocLayoutConfig,
    preprocessor: PpDocLayoutPreprocessorConfig,
    detector: PpDocLayoutDetector<B>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpPostprocessOptions {
    score_threshold: f32,
    nms_threshold: Option<f32>,
    max_detections: usize,
}

impl<B> PpEncoderInputProjection<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        input_channels: &[usize],
        output_channels: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Self::from_safetensors_with_prefix(
            tensors,
            "model.encoder_input_proj",
            input_channels,
            output_channels,
            device,
        )
    }

    pub(crate) fn from_safetensors_with_prefix(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        input_channels: &[usize],
        output_channels: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let mut projections = Vec::with_capacity(input_channels.len());
        for (index, channels) in input_channels.iter().copied().enumerate() {
            projections.push(PpConv1x1BatchNorm::from_safetensors(
                tensors,
                &format!("{prefix}.{index}.0"),
                &format!("{prefix}.{index}.1"),
                channels,
                output_channels,
                device,
            )?);
        }

        Ok(Self { projections })
    }

    pub(crate) fn forward(
        &self,
        features: Vec<Tensor<B, 4>>,
    ) -> Result<Vec<Tensor<B, 4>>> {
        if features.len() != self.projections.len() {
            bail!(
                "PP-DocLayoutV3 expected {} feature maps, got {}",
                self.projections.len(),
                features.len()
            );
        }

        Ok(features
            .into_iter()
            .zip(self.projections.iter())
            .map(|(feature, projection)| projection.forward(feature))
            .collect())
    }
}

impl<B> PpConvBatchNorm<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        conv_prefix: &str,
        norm_prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: PaddingConfig2d,
        activation: PpActivation,
        device: &B::Device,
    ) -> Result<Self> {
        Self::from_safetensors_grouped(
            tensors,
            conv_prefix,
            norm_prefix,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            1,
            activation,
            device,
        )
    }

    pub(crate) fn from_safetensors_grouped(
        tensors: &SafeTensors<'_>,
        conv_prefix: &str,
        norm_prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: PaddingConfig2d,
        groups: usize,
        activation: PpActivation,
        device: &B::Device,
    ) -> Result<Self> {
        let conv = Conv2d {
            weight: Param::from_tensor(read_conv2d_weight(
                tensors,
                &format!("{conv_prefix}.weight"),
                output_channels,
                input_channels / groups,
                kernel_size,
                device,
            )?),
            bias: None,
            stride,
            kernel_size,
            dilation: [1, 1],
            groups,
            padding,
        };

        Ok(Self {
            conv,
            norm_weight: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.weight"),
                &[output_channels],
                device,
            )?,
            norm_bias: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.bias"),
                &[output_channels],
                device,
            )?,
            running_mean: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.running_mean"),
                &[output_channels],
                device,
            )?,
            running_var: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.running_var"),
                &[output_channels],
                device,
            )?,
            epsilon: 1.0e-5,
            activation,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let normalized = batch_norm_inference(
            self.conv.forward(x),
            self.norm_weight.clone(),
            self.norm_bias.clone(),
            self.running_mean.clone(),
            self.running_var.clone(),
            self.epsilon,
        );

        match self.activation {
            PpActivation::Identity => normalized,
            PpActivation::Relu => relu(normalized),
            PpActivation::Silu => silu(normalized),
        }
    }
}

impl<B> PpHgnetStem<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        let prefix = "model.backbone.model.embedder";
        Ok(Self {
            stem1: load_hgnet_conv(
                tensors,
                prefix,
                "stem1",
                3,
                32,
                [3, 3],
                [2, 2],
                device,
            )?,
            stem2a: load_hgnet_conv(
                tensors,
                prefix,
                "stem2a",
                32,
                16,
                [2, 2],
                [1, 1],
                device,
            )?,
            stem2b: load_hgnet_conv(
                tensors,
                prefix,
                "stem2b",
                16,
                32,
                [2, 2],
                [1, 1],
                device,
            )?,
            stem3: load_hgnet_conv(
                tensors,
                prefix,
                "stem3",
                64,
                32,
                [3, 3],
                [2, 2],
                device,
            )?,
            stem4: load_hgnet_conv(
                tensors,
                prefix,
                "stem4",
                32,
                48,
                [1, 1],
                [1, 1],
                device,
            )?,
        })
    }

    pub(crate) fn forward(&self, pixel_values: Tensor<B, 4>) -> Tensor<B, 4> {
        let embedding = pad_right_bottom(self.stem1.forward(pixel_values));
        let stem2 = self
            .stem2b
            .forward(pad_right_bottom(self.stem2a.forward(embedding.clone())));
        let pooled =
            max_pool2d(embedding, [2, 2], [1, 1], [0, 0], [1, 1], true);
        let fused = Tensor::cat(vec![pooled, stem2], 1);

        self.stem4.forward(self.stem3.forward(fused))
    }
}

impl<B> PpHgnetBasicLayer<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        input_channels: usize,
        middle_channels: usize,
        output_channels: usize,
        layer_count: usize,
        kernel_size: [usize; 2],
        residual: bool,
        light_block: bool,
        device: &B::Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(layer_count);
        for layer_index in 0..layer_count {
            let layer_input_channels = if layer_index == 0 {
                input_channels
            } else {
                middle_channels
            };
            let layer_prefix = format!("{prefix}.layers.{layer_index}");
            if light_block {
                layers.push(PpHgnetLayer::Light {
                    pointwise: load_conv_layer(
                        tensors,
                        &format!("{layer_prefix}.conv1"),
                        layer_input_channels,
                        middle_channels,
                        [1, 1],
                        [1, 1],
                        1,
                        PpActivation::Identity,
                        device,
                    )?,
                    depthwise: load_conv_layer(
                        tensors,
                        &format!("{layer_prefix}.conv2"),
                        middle_channels,
                        middle_channels,
                        kernel_size,
                        [1, 1],
                        middle_channels,
                        PpActivation::Relu,
                        device,
                    )?,
                });
            } else {
                layers.push(PpHgnetLayer::Conv(load_conv_layer(
                    tensors,
                    &layer_prefix,
                    layer_input_channels,
                    middle_channels,
                    kernel_size,
                    [1, 1],
                    1,
                    PpActivation::Relu,
                    device,
                )?));
            }
        }

        let aggregation_input_channels =
            input_channels + layer_count * middle_channels;
        Ok(Self {
            layers,
            aggregation_squeeze: load_conv_layer(
                tensors,
                &format!("{prefix}.aggregation.0"),
                aggregation_input_channels,
                output_channels / 2,
                [1, 1],
                [1, 1],
                1,
                PpActivation::Relu,
                device,
            )?,
            aggregation_excitation: load_conv_layer(
                tensors,
                &format!("{prefix}.aggregation.1"),
                output_channels / 2,
                output_channels,
                [1, 1],
                [1, 1],
                1,
                PpActivation::Relu,
                device,
            )?,
            residual,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = x.clone();
        let mut outputs = vec![x.clone()];
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(hidden);
            outputs.push(hidden.clone());
        }

        let aggregated = self
            .aggregation_excitation
            .forward(self.aggregation_squeeze.forward(Tensor::cat(outputs, 1)));
        if self.residual {
            return aggregated + identity;
        }

        aggregated
    }
}

impl<B> PpHgnetLayer<B>
where
    B: Backend<FloatElem = f32>,
{
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Conv(layer) => layer.forward(x),
            Self::Light {
                pointwise,
                depthwise,
            } => depthwise.forward(pointwise.forward(x)),
        }
    }
}

impl<B> PpHgnetStage<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn stage0_from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Self::stage_from_safetensors(
            tensors,
            0,
            48,
            48,
            128,
            1,
            6,
            false,
            false,
            [3, 3],
            device,
        )
    }

    pub(crate) fn stage1_from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Self::stage_from_safetensors(
            tensors,
            1,
            128,
            96,
            512,
            1,
            6,
            true,
            false,
            [3, 3],
            device,
        )
    }

    pub(crate) fn stage2_from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Self::stage_from_safetensors(
            tensors,
            2,
            512,
            192,
            1024,
            3,
            6,
            true,
            true,
            [5, 5],
            device,
        )
    }

    pub(crate) fn stage3_from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Self::stage_from_safetensors(
            tensors,
            3,
            1024,
            384,
            2048,
            1,
            6,
            true,
            true,
            [5, 5],
            device,
        )
    }

    pub(crate) fn stage_from_safetensors(
        tensors: &SafeTensors<'_>,
        stage_index: usize,
        input_channels: usize,
        middle_channels: usize,
        output_channels: usize,
        block_count: usize,
        layer_count: usize,
        downsample: bool,
        light_block: bool,
        kernel_size: [usize; 2],
        device: &B::Device,
    ) -> Result<Self> {
        let stage_prefix =
            format!("model.backbone.model.encoder.stages.{stage_index}");
        let downsample_layer = if downsample {
            Some(load_conv_layer(
                tensors,
                &format!("{stage_prefix}.downsample"),
                input_channels,
                input_channels,
                [3, 3],
                [2, 2],
                input_channels,
                PpActivation::Identity,
                device,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(block_count);
        for block_index in 0..block_count {
            blocks.push(PpHgnetBasicLayer::from_safetensors(
                tensors,
                &format!("{stage_prefix}.blocks.{block_index}"),
                if block_index == 0 {
                    input_channels
                } else {
                    output_channels
                },
                middle_channels,
                output_channels,
                layer_count,
                kernel_size,
                block_index != 0,
                light_block,
                device,
            )?);
        }

        Ok(Self {
            downsample: downsample_layer,
            blocks,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut hidden = match &self.downsample {
            Some(downsample) => downsample.forward(x),
            None => x,
        };
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        hidden
    }
}

impl<B> PpHgnetBackbone<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            stem: PpHgnetStem::from_safetensors(tensors, device)?,
            stage0: PpHgnetStage::stage0_from_safetensors(tensors, device)?,
            stage1: PpHgnetStage::stage1_from_safetensors(tensors, device)?,
            stage2: PpHgnetStage::stage2_from_safetensors(tensors, device)?,
            stage3: PpHgnetStage::stage3_from_safetensors(tensors, device)?,
        })
    }

    pub(crate) fn forward(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> Vec<Tensor<B, 4>> {
        let stem = self.stem.forward(pixel_values);
        let stage0 = self.stage0.forward(stem);
        let stage1 = self.stage1.forward(stage0);
        let stage2 = self.stage2.forward(stage1.clone());
        let stage3 = self.stage3.forward(stage2.clone());

        vec![stage1, stage2, stage3]
    }
}

impl<B> PpBackboneFeatureProjector<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            backbone: PpHgnetBackbone::from_safetensors(tensors, device)?,
            projection: PpEncoderInputProjection::from_safetensors(
                tensors,
                &[512, 1024, 2048],
                256,
                device,
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> Result<Vec<Tensor<B, 4>>> {
        let features = self.backbone.forward(pixel_values);
        self.projection.forward(features)
    }
}

impl<B> PpHybridEncoderConvs<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        let mut lateral_convs = Vec::with_capacity(2);
        let mut downsample_convs = Vec::with_capacity(2);
        let mut fpn_blocks = Vec::with_capacity(2);
        let mut pan_blocks = Vec::with_capacity(2);
        for index in 0..2 {
            lateral_convs.push(load_conv_norm_layer(
                tensors,
                &format!("model.encoder.lateral_convs.{index}"),
                256,
                256,
                [1, 1],
                [1, 1],
                PpActivation::Silu,
                device,
            )?);
            downsample_convs.push(load_conv_norm_layer(
                tensors,
                &format!("model.encoder.downsample_convs.{index}"),
                256,
                256,
                [3, 3],
                [2, 2],
                PpActivation::Silu,
                device,
            )?);
            fpn_blocks.push(PpCspRepLayer::from_safetensors(
                tensors,
                &format!("model.encoder.fpn_blocks.{index}"),
                device,
            )?);
            pan_blocks.push(PpCspRepLayer::from_safetensors(
                tensors,
                &format!("model.encoder.pan_blocks.{index}"),
                device,
            )?);
        }

        Ok(Self {
            aifi: PpAifiLayer::from_safetensors(
                tensors,
                "model.encoder.encoder.0.layers.0",
                device,
            )?,
            lateral_convs,
            downsample_convs,
            fpn_blocks,
            pan_blocks,
        })
    }

    pub(crate) fn forward(
        &self,
        mut features: Vec<Tensor<B, 4>>,
    ) -> Vec<Tensor<B, 4>> {
        features[2] = self.aifi.forward(features[2].clone());
        let mut fpn_features =
            vec![features.pop().expect("top feature must exist")];
        for index in 0..2 {
            let backbone_feature =
                features.pop().expect("backbone feature must exist");
            let top = self.lateral_convs[index].forward(
                fpn_features.pop().expect("top FPN feature must exist"),
            );
            let [_batch, _channels, height, width] = backbone_feature.dims();
            let fused = Tensor::cat(
                vec![
                    upsample_nearest_to(top.clone(), height, width),
                    backbone_feature,
                ],
                1,
            );
            fpn_features.push(top);
            fpn_features.push(self.fpn_blocks[index].forward(fused));
        }
        fpn_features.reverse();

        let mut pan_features = vec![fpn_features.remove(0)];
        for index in 0..2 {
            let downsampled = self.downsample_convs[index].forward(
                pan_features.last().expect("PAN feature must exist").clone(),
            );
            let fused =
                Tensor::cat(vec![downsampled, fpn_features.remove(0)], 1);
            pan_features.push(self.pan_blocks[index].forward(fused));
        }

        pan_features
    }
}

impl<B> PpRepVggBlock<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            conv1: load_conv_norm_layer(
                tensors,
                &format!("{prefix}.conv1"),
                256,
                256,
                [3, 3],
                [1, 1],
                PpActivation::Identity,
                device,
            )?,
            conv2: load_conv_norm_layer(
                tensors,
                &format!("{prefix}.conv2"),
                256,
                256,
                [1, 1],
                [1, 1],
                PpActivation::Identity,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        silu(self.conv1.forward(x.clone()) + self.conv2.forward(x))
    }
}

impl<B> PpCspRepLayer<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        let mut bottlenecks = Vec::with_capacity(3);
        for index in 0..3 {
            bottlenecks.push(PpRepVggBlock::from_safetensors(
                tensors,
                &format!("{prefix}.bottlenecks.{index}"),
                device,
            )?);
        }

        Ok(Self {
            conv1: load_conv_norm_layer(
                tensors,
                &format!("{prefix}.conv1"),
                512,
                256,
                [1, 1],
                [1, 1],
                PpActivation::Silu,
                device,
            )?,
            conv2: load_conv_norm_layer(
                tensors,
                &format!("{prefix}.conv2"),
                512,
                256,
                [1, 1],
                [1, 1],
                PpActivation::Silu,
                device,
            )?,
            bottlenecks,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut left = self.conv1.forward(x.clone());
        for bottleneck in &self.bottlenecks {
            left = bottleneck.forward(left);
        }
        left + self.conv2.forward(x)
    }
}

impl<B> PpLinear<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        input_features: usize,
        output_features: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: read_linear_weight(
                tensors,
                &format!("{prefix}.weight"),
                input_features,
                output_features,
                device,
            )?,
            bias: read_f32_tensor(
                tensors,
                &format!("{prefix}.bias"),
                &[output_features],
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, sequence, input_features] = x.dims();
        let flattened = x.reshape([batch * sequence, input_features]);
        let projected = safe_matmul(flattened, self.weight.clone())
            + self.bias.clone().unsqueeze();
        let [_rows, output_features] = projected.dims();

        projected.reshape([batch, sequence, output_features])
    }
}

impl<B> PpMlp<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        dims: &[(usize, usize)],
        device: &B::Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(dims.len());
        for (index, (input_features, output_features)) in
            dims.iter().copied().enumerate()
        {
            layers.push(PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.layers.{index}"),
                input_features,
                output_features,
                device,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index + 1 < self.layers.len() {
                x = relu(x);
            }
        }
        x
    }
}

impl<B> PpLayerNorm<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: read_f32_tensor(
                tensors,
                &format!("{prefix}.weight"),
                &[size],
                device,
            )?,
            bias: read_f32_tensor(
                tensors,
                &format!("{prefix}.bias"),
                &[size],
                device,
            )?,
            epsilon: 1.0e-5,
        })
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, _sequence, hidden] = x.dims();
        let mean = x.clone().mean_dim(2);
        let centered = x - mean;
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered
            * (variance + self.epsilon).sqrt().recip()
            * self.weight.clone().reshape([1, 1, hidden])
            + self.bias.clone().reshape([1, 1, hidden])
    }
}

impl<B> PpAifiLayer<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn.q_proj"),
                256,
                256,
                device,
            )?,
            k_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn.k_proj"),
                256,
                256,
                device,
            )?,
            v_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn.v_proj"),
                256,
                256,
                device,
            )?,
            out_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn.out_proj"),
                256,
                256,
                device,
            )?,
            self_attn_layer_norm: PpLayerNorm::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn_layer_norm"),
                256,
                device,
            )?,
            fc1: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.fc1"),
                256,
                1024,
                device,
            )?,
            fc2: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.fc2"),
                1024,
                256,
                device,
            )?,
            final_layer_norm: PpLayerNorm::from_safetensors(
                tensors,
                &format!("{prefix}.final_layer_norm"),
                256,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let hidden = x.swap_dims(1, 3).swap_dims(1, 2).reshape([
            batch,
            height * width,
            channels,
        ]);
        // RT-DETR's AIFI 2D-sincos positional embedding is a runtime-computed
        // buffer (not a stored weight); compute it natively here. Verified
        // identical (max abs diff ~6e-8) to PaddlePaddle's baked `eager_tmp_0`.
        let position =
            aifi_position_embedding::<B>(height, width, &hidden.device());
        let residual = hidden.clone();
        let attended = self.self_attention(hidden, position);
        let hidden = self.self_attn_layer_norm.forward(residual + attended);
        let residual = hidden.clone();
        let mlp = self.fc2.forward(gelu(self.fc1.forward(hidden)));
        self.final_layer_norm
            .forward(residual + mlp)
            .reshape([batch, height, width, channels])
            .swap_dims(1, 2)
            .swap_dims(1, 3)
    }

    fn self_attention(
        &self,
        hidden: Tensor<B, 3>,
        position: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, sequence, _hidden] = hidden.dims();
        let heads = 8;
        let head_dim = 32;
        let query_key_input = hidden.clone() + position;
        let q = self
            .q_proj
            .forward(query_key_input.clone())
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(query_key_input)
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(hidden)
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2);
        let scores =
            safe_matmul(q, k.swap_dims(2, 3)) / (head_dim as f64).sqrt();
        let context = safe_matmul(softmax(scores, 3), v)
            .swap_dims(1, 2)
            .reshape([batch, sequence, 256]);

        self.out_proj.forward(context)
    }
}

impl<B> PpEncoderDetectionHead<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            decoder_input_proj:
                PpEncoderInputProjection::from_safetensors_with_prefix(
                    tensors,
                    "model.decoder_input_proj",
                    &[256, 256, 256],
                    256,
                    device,
                )?,
            enc_output: PpLinear::from_safetensors(
                tensors,
                "model.enc_output.0",
                256,
                256,
                device,
            )?,
            enc_output_norm: PpLayerNorm::from_safetensors(
                tensors,
                "model.enc_output.1",
                256,
                device,
            )?,
            enc_score_head: PpLinear::from_safetensors(
                tensors,
                "model.enc_score_head",
                256,
                25,
                device,
            )?,
            enc_bbox_head: PpMlp::from_safetensors(
                tensors,
                "model.enc_bbox_head",
                &[(256, 256), (256, 256), (256, 4)],
                device,
            )?,
        })
    }

    fn forward(
        &self,
        features: Vec<Tensor<B, 4>>,
    ) -> Result<(
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Vec<(usize, usize)>,
    )> {
        let (anchors, valid_mask) =
            generate_encoder_anchors(&features, &features[0].device())?;
        let projected = self.decoder_input_proj.forward(features)?;
        let spatial_shapes = projected
            .iter()
            .map(|feature| {
                let [_batch, _channels, height, width] = feature.dims();
                (height, width)
            })
            .collect();
        let source_flatten = flatten_feature_maps(projected);
        let memory = source_flatten.clone() * valid_mask.clone();
        let encoded = self
            .enc_output_norm
            .forward(self.enc_output.forward(memory.clone()));
        let encoder_scores = self.enc_score_head.forward(encoded.clone());
        Ok((
            encoder_scores,
            self.enc_bbox_head.forward(encoded.clone()) + anchors,
            encoded,
            source_flatten,
            spatial_shapes,
        ))
    }
}

impl<B> PpDecoderAttention<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.q_proj"),
                256,
                256,
                device,
            )?,
            k_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.k_proj"),
                256,
                256,
                device,
            )?,
            v_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.v_proj"),
                256,
                256,
                device,
            )?,
            out_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.out_proj"),
                256,
                256,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        query_pos: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, sequence, _hidden] = hidden.dims();
        let heads = 8;
        let head_dim = 32;
        let query = hidden.clone() + query_pos.clone();
        let key = hidden.clone() + query_pos;
        let q = self
            .q_proj
            .forward(query)
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2);
        let k = self
            .k_proj
            .forward(key)
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(hidden)
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2);
        let scores =
            safe_matmul(q, k.swap_dims(2, 3)) / (head_dim as f64).sqrt();
        let context = safe_matmul(softmax(scores, 3), v)
            .swap_dims(1, 2)
            .reshape([batch, sequence, 256]);

        self.out_proj.forward(context)
    }
}

impl<B> PpDecoderCrossAttention<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            sampling_offsets: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.sampling_offsets"),
                256,
                192,
                device,
            )?,
            attention_weights: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.attention_weights"),
                256,
                96,
                device,
            )?,
            value_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.value_proj"),
                256,
                256,
                device,
            )?,
            output_proj: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.output_proj"),
                256,
                256,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        query_pos: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        reference_boxes: Tensor<B, 3>,
        spatial_shapes: &[(usize, usize)],
    ) -> Tensor<B, 3> {
        let [batch, queries, _hidden] = hidden.dims();
        let [_memory_batch, sequence, _memory_hidden] = memory.dims();
        let heads = 8;
        let head_dim = 32;
        let levels = spatial_shapes.len();
        let device = hidden.device();
        let value = self
            .value_proj
            .forward(memory)
            .reshape([batch, sequence, heads, head_dim]);
        let query = hidden + query_pos;
        let offsets = self
            .sampling_offsets
            .forward(query.clone())
            .reshape([batch, queries, heads, levels, 4, 2]);
        let weights = self.attention_weights.forward(query).reshape([
            batch,
            queries,
            heads,
            levels * 4,
        ]);
        let context = deformable_attention_context(
            value,
            offsets,
            weights,
            reference_boxes,
            spatial_shapes,
        )
        .unwrap_or_else(|_| {
            Tensor::zeros([batch, queries, heads * head_dim], &device)
        });

        self.output_proj.forward(context)
    }
}

impl<B> PpDecoderLayer<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: PpDecoderAttention::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn"),
                device,
            )?,
            self_attn_layer_norm: PpLayerNorm::from_safetensors(
                tensors,
                &format!("{prefix}.self_attn_layer_norm"),
                256,
                device,
            )?,
            encoder_attn: PpDecoderCrossAttention::from_safetensors(
                tensors,
                &format!("{prefix}.encoder_attn"),
                device,
            )?,
            encoder_attn_layer_norm: PpLayerNorm::from_safetensors(
                tensors,
                &format!("{prefix}.encoder_attn_layer_norm"),
                256,
                device,
            )?,
            fc1: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.fc1"),
                256,
                1024,
                device,
            )?,
            fc2: PpLinear::from_safetensors(
                tensors,
                &format!("{prefix}.fc2"),
                1024,
                256,
                device,
            )?,
            final_layer_norm: PpLayerNorm::from_safetensors(
                tensors,
                &format!("{prefix}.final_layer_norm"),
                256,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        query_pos: Tensor<B, 3>,
        memory: Tensor<B, 3>,
        reference_boxes: Tensor<B, 3>,
        spatial_shapes: &[(usize, usize)],
    ) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let hidden = self.self_attn_layer_norm.forward(
            residual + self.self_attn.forward(hidden, query_pos.clone()),
        );
        let residual = hidden.clone();
        let hidden = self.encoder_attn_layer_norm.forward(
            residual
                + self.encoder_attn.forward(
                    hidden,
                    query_pos,
                    memory,
                    reference_boxes,
                    spatial_shapes,
                ),
        );
        let residual = hidden.clone();
        let mlp = self.fc2.forward(relu(self.fc1.forward(hidden)));

        self.final_layer_norm.forward(residual + mlp)
    }
}

impl<B> PpDocLayoutDecoder<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(6);
        for index in 0..6 {
            layers.push(PpDecoderLayer::from_safetensors(
                tensors,
                &format!("model.decoder.layers.{index}"),
                device,
            )?);
        }
        let mut order_head = Vec::with_capacity(6);
        for index in 0..6 {
            order_head.push(PpLinear::from_safetensors(
                tensors,
                &format!("model.decoder_order_head.{index}"),
                256,
                256,
                device,
            )?);
        }

        Ok(Self {
            query_pos_head: PpMlp::from_safetensors(
                tensors,
                "model.decoder.query_pos_head",
                &[(4, 512), (512, 256)],
                device,
            )?,
            layers,
            order_head,
            decoder_norm: PpLayerNorm::from_safetensors(
                tensors,
                "model.decoder_norm",
                256,
                device,
            )?,
            global_pointer: PpLinear::from_safetensors(
                tensors,
                "model.decoder_global_pointer.dense",
                256,
                128,
                device,
            )?,
        })
    }

    fn forward(
        &self,
        output_memory: Tensor<B, 3>,
        encoder_hidden_states: Tensor<B, 3>,
        encoder_scores: Tensor<B, 3>,
        proposal_boxes: Tensor<B, 3>,
        bbox_head: &PpMlp<B>,
        score_head: &PpLinear<B>,
        spatial_shapes: &[(usize, usize)],
        num_queries: usize,
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>)> {
        let [batch, sequence, hidden] = output_memory.dims();
        if batch != 1 {
            bail!("PP-DocLayoutV3 decoder currently expects batch size 1");
        }
        if sequence < num_queries {
            bail!(
                "PP-DocLayoutV3 decoder needs at least {num_queries} encoder proposals, got {sequence}"
            );
        }

        let indices = topk_proposal_indices(encoder_scores, num_queries)?;
        let query = gather_sequence(output_memory, &indices, hidden)?;
        let mut reference =
            sigmoid_tensor(gather_sequence(proposal_boxes, &indices, 4)?);
        let mut decoded = query;
        for layer in &self.layers {
            let query_pos = self.query_pos_head.forward(reference.clone());
            decoded = layer.forward(
                decoded,
                query_pos,
                encoder_hidden_states.clone(),
                reference.clone(),
                spatial_shapes,
            );
            reference = sigmoid_tensor(
                bbox_head.forward(decoded.clone())
                    + inverse_sigmoid_tensor(reference),
            );
        }
        let normalized = self.decoder_norm.forward(decoded);
        let class_scores = score_head.forward(normalized.clone());
        let order_features = self.global_pointer.forward(
            self.order_head[self.order_head.len() - 1].forward(normalized),
        );

        Ok((reference, class_scores, order_features))
    }
}

impl<B> PpDocLayoutDetector<B>
where
    B: Backend<FloatElem = f32>,
{
    fn from_safetensors(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            backbone: PpBackboneFeatureProjector::from_safetensors(
                tensors, device,
            )?,
            encoder: PpHybridEncoderConvs::from_safetensors(tensors, device)?,
            detection_head: PpEncoderDetectionHead::from_safetensors(
                tensors, device,
            )?,
            decoder: PpDocLayoutDecoder::from_safetensors(tensors, device)?,
        })
    }

    fn forward(
        &self,
        pixel_values: Tensor<B, 4>,
    ) -> Result<PpDocLayoutRawOutput<B>> {
        let projected = self.backbone.forward(pixel_values)?;
        let pan_features = self.encoder.forward(projected);
        let (
            encoder_scores,
            encoder_boxes,
            output_memory,
            encoder_hidden_states,
            spatial_shapes,
        ) = self.detection_head.forward(pan_features)?;
        let num_queries = 300.min(encoder_boxes.dims()[1]);
        let (reference_boxes, decoder_scores, order_features) =
            self.decoder.forward(
                output_memory,
                encoder_hidden_states,
                encoder_scores,
                encoder_boxes,
                &self.detection_head.enc_bbox_head,
                &self.detection_head.enc_score_head,
                &spatial_shapes,
                num_queries,
            )?;
        Ok(PpDocLayoutRawOutput {
            scores: decoder_scores,
            boxes: reference_boxes,
            order_features,
        })
    }
}

fn load_hgnet_conv<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    name: &str,
    input_channels: usize,
    output_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    device: &B::Device,
) -> Result<PpConvBatchNorm<B>> {
    load_conv_layer(
        tensors,
        &format!("{prefix}.{name}"),
        input_channels,
        output_channels,
        kernel_size,
        stride,
        1,
        PpActivation::Relu,
        device,
    )
}

fn load_conv_layer<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    groups: usize,
    activation: PpActivation,
    device: &B::Device,
) -> Result<PpConvBatchNorm<B>> {
    PpConvBatchNorm::from_safetensors_grouped(
        tensors,
        &format!("{prefix}.convolution"),
        &format!("{prefix}.normalization"),
        input_channels,
        output_channels,
        kernel_size,
        stride,
        PaddingConfig2d::Explicit(
            (kernel_size[0] - 1) / 2,
            (kernel_size[1] - 1) / 2,
            (kernel_size[0] - 1) / 2,
            (kernel_size[1] - 1) / 2,
        ),
        groups,
        activation,
        device,
    )
}

fn load_conv_norm_layer<B: Backend<FloatElem = f32>>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    input_channels: usize,
    output_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    activation: PpActivation,
    device: &B::Device,
) -> Result<PpConvBatchNorm<B>> {
    PpConvBatchNorm::from_safetensors(
        tensors,
        &format!("{prefix}.conv"),
        &format!("{prefix}.norm"),
        input_channels,
        output_channels,
        kernel_size,
        stride,
        PaddingConfig2d::Explicit(
            (kernel_size[0] - 1) / 2,
            (kernel_size[1] - 1) / 2,
            (kernel_size[0] - 1) / 2,
            (kernel_size[1] - 1) / 2,
        ),
        activation,
        device,
    )
}

fn upsample_nearest_to<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 4>,
    height: usize,
    width: usize,
) -> Tensor<B, 4> {
    interpolate(
        x,
        [height, width],
        InterpolateOptions::new(InterpolateMode::Nearest),
    )
}

fn pad_right_bottom<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 4>,
) -> Tensor<B, 4> {
    x.pad([(0, 0), (0, 0), (0, 1), (0, 1)], PadMode::Constant(0.0))
}

impl<B> PpConv1x1BatchNorm<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn from_safetensors(
        tensors: &SafeTensors<'_>,
        conv_prefix: &str,
        norm_prefix: &str,
        input_channels: usize,
        output_channels: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: read_conv1x1_weight(
                tensors,
                &format!("{conv_prefix}.weight"),
                output_channels,
                input_channels,
                device,
            )?,
            bias: Tensor::zeros([output_channels], device),
            norm_weight: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.weight"),
                &[output_channels],
                device,
            )?,
            norm_bias: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.bias"),
                &[output_channels],
                device,
            )?,
            running_mean: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.running_mean"),
                &[output_channels],
                device,
            )?,
            running_var: read_f32_tensor(
                tensors,
                &format!("{norm_prefix}.running_var"),
                &[output_channels],
                device,
            )?,
            epsilon: 1.0e-5,
        })
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let flattened = x
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch * height * width, channels]);
        let projected = safe_matmul(flattened, self.weight.clone())
            + self.bias.clone().unsqueeze();
        let normalized = (projected - self.running_mean.clone().unsqueeze())
            * (self.running_var.clone().unsqueeze() + self.epsilon)
                .sqrt()
                .recip()
            * self.norm_weight.clone().unsqueeze()
            + self.norm_bias.clone().unsqueeze();
        let [_rows, output_channels] = normalized.dims();

        normalized
            .reshape([batch, height, width, output_channels])
            .swap_dims(1, 2)
            .swap_dims(1, 3)
    }
}

async fn load_pp_doclayout_files(
    cache_dir: Option<PathBuf>,
) -> Result<PpDocLayoutFiles> {
    let mut builder = ApiBuilder::new();
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }
    let api = builder.build()?;
    let repo =
        api.repo(Repo::new(PP_DOCLAYOUT_REPO_ID.to_string(), RepoType::Model));

    let weights_path = match std::env::var_os(PP_DOCLAYOUT_WEIGHTS_ENV) {
        Some(path) => PathBuf::from(path),
        None => repo.get(PP_DOCLAYOUT_WEIGHTS).await?,
    };

    Ok(PpDocLayoutFiles {
        config_path: repo.get(PP_DOCLAYOUT_CONFIG).await?,
        preprocessor_path: repo.get(PP_DOCLAYOUT_PREPROCESSOR).await?,
        weights_path,
    })
}

pub(crate) async fn load_pp_doclayout_runtime<B>(
    device: &B::Device,
    cache_dir: Option<PathBuf>,
) -> Result<PpDocLayoutRuntime<B>>
where
    B: Backend<FloatElem = f32>,
{
    let files = load_pp_doclayout_files(cache_dir).await?;
    let (config, preprocessor) = read_pp_doclayout_config(&files)?;
    let bytes = std::fs::read(&files.weights_path).with_context(|| {
        format!("failed to read {}", files.weights_path.display())
    })?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    validate_pp_doclayout_weights(&tensors)?;
    let detector = PpDocLayoutDetector::from_safetensors(&tensors, device)?;

    Ok(PpDocLayoutRuntime {
        config,
        preprocessor,
        detector,
    })
}

impl<B> PpDocLayoutRuntime<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn detect_image(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<Vec<LayoutDetection>> {
        let input = preprocess_layout_image(image, &self.preprocessor)?;
        let channels = input.channels;
        let height = input.height;
        let width = input.width;
        let tensor = Tensor::from_data(
            TensorData::new(input.values.clone(), [1, channels, height, width]),
            device,
        );
        let raw = self.detector.forward(tensor)?;

        postprocess_encoder_proposals(
            raw,
            &self.config.id2label,
            input.original_width,
            input.original_height,
            PpPostprocessOptions::default(),
        )
    }
}

fn read_pp_doclayout_config(
    files: &PpDocLayoutFiles,
) -> Result<(PpDocLayoutConfig, PpDocLayoutPreprocessorConfig)> {
    let config = serde_json::from_slice::<PpDocLayoutConfig>(
        &std::fs::read(&files.config_path).with_context(|| {
            format!("failed to read {}", files.config_path.display())
        })?,
    )?;
    let preprocessor = serde_json::from_slice::<PpDocLayoutPreprocessorConfig>(
        &std::fs::read(&files.preprocessor_path).with_context(|| {
            format!("failed to read {}", files.preprocessor_path.display())
        })?,
    )?;

    validate_pp_doclayout_config(&config, &preprocessor)?;
    Ok((config, preprocessor))
}

fn validate_pp_doclayout_weights(tensors: &SafeTensors<'_>) -> Result<()> {
    let required = [
        "model.backbone.model.embedder.stem1.convolution.weight",
        "model.encoder_input_proj.0.0.weight",
        "model.encoder.lateral_convs.0.conv.weight",
        "model.enc_bbox_head.layers.0.weight",
        "model.decoder.query_pos_head.layers.0.weight",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "model.decoder.layers.0.encoder_attn.sampling_offsets.weight",
        "model.decoder_norm.weight",
        "model.decoder_order_head.0.weight",
        "model.decoder_global_pointer.dense.weight",
        "model.denoising_class_embed.weight",
    ];

    for name in required {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("missing PP-DocLayoutV3 tensor {name}"))?;
        if tensor.dtype() != Dtype::F32 {
            bail!("PP-DocLayoutV3 tensor {name} must be float32");
        }
    }

    Ok(())
}

fn validate_pp_doclayout_config(
    config: &PpDocLayoutConfig,
    preprocessor: &PpDocLayoutPreprocessorConfig,
) -> Result<()> {
    if config.model_type != "pp_doclayout_v3" {
        bail!("unsupported layout model type {}", config.model_type);
    }
    if !config
        .architectures
        .iter()
        .any(|name| name == "PPDocLayoutV3ForObjectDetection")
    {
        bail!("PP-DocLayoutV3 object detection architecture missing");
    }
    if config.d_model != 256 || config.num_queries != 300 {
        bail!("unexpected PP-DocLayoutV3 dimensions");
    }
    if config.decoder_layers != 6
        || config.decoder_attention_heads != 8
        || config.decoder_n_points != 4
    {
        bail!("unexpected PP-DocLayoutV3 decoder config");
    }
    if config.feature_strides != [8, 16, 32] {
        bail!("unexpected PP-DocLayoutV3 feature strides");
    }
    if preprocessor.do_resize
        && (preprocessor.size.height == 0 || preprocessor.size.width == 0)
    {
        bail!("PP-DocLayoutV3 resize dimensions must be positive");
    }

    Ok(())
}

pub(crate) fn sort_detections_by_order(
    mut detections: Vec<LayoutDetection>,
) -> Vec<LayoutDetection> {
    detections.sort_by(|left, right| {
        left.order
            .cmp(&right.order)
            .then_with(|| left.bbox[1].total_cmp(&right.bbox[1]))
            .then_with(|| left.bbox[0].total_cmp(&right.bbox[0]))
    });
    detections
}

fn preprocess_layout_image(
    image: &DynamicImage,
    config: &PpDocLayoutPreprocessorConfig,
) -> Result<LayoutInput> {
    let (original_width, original_height) = image.dimensions();
    if original_width == 0 || original_height == 0 {
        bail!("layout input image cannot be empty");
    }
    if config.image_mean.len() < 3 || config.image_std.len() < 3 {
        bail!("layout image normalization config must have three channels");
    }

    // PaddleX resizes the layout input with cv2 INTER_CUBIC; match it
    // byte-for-byte (see `crate::ml::imageproc`).
    let rgb = if config.do_resize {
        crate::ml::imageproc::resize_cubic_cv2(
            image,
            config.size.width,
            config.size.height,
        )
    } else {
        image.to_rgb8()
    };
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let mut values = vec![0.0; 3 * height * width];

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                let scaled = f32::from(pixel[channel]) / 255.0;
                values[channel * height * width + y * width + x] = (scaled
                    - config.image_mean[channel])
                    / config.image_std[channel];
            }
        }
    }

    Ok(LayoutInput {
        values,
        channels: 3,
        height,
        width,
        original_height,
        original_width,
    })
}

impl Default for PpPostprocessOptions {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            nms_threshold: None,
            max_detections: 300,
        }
    }
}

pub(crate) fn postprocess_encoder_proposals<B: Backend<FloatElem = f32>>(
    output: PpDocLayoutRawOutput<B>,
    labels: &std::collections::BTreeMap<String, String>,
    original_width: u32,
    original_height: u32,
    options: PpPostprocessOptions,
) -> Result<Vec<LayoutDetection>> {
    let [batch, proposals, classes] = output.scores.dims();
    if batch != 1 {
        bail!("PP-DocLayoutV3 postprocess expects batch size 1");
    }
    let [box_batch, box_proposals, box_dims] = output.boxes.dims();
    if box_batch != batch || box_proposals != proposals || box_dims != 4 {
        bail!("PP-DocLayoutV3 score and box shapes do not match");
    }
    if classes == 0 {
        bail!("PP-DocLayoutV3 postprocess expects class scores");
    }

    let scores = output.scores.into_data().to_vec::<f32>()?;
    let boxes = output.boxes.into_data().to_vec::<f32>()?;
    let orders = reading_order_ranks(output.order_features)?;
    let mut candidates = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(index, logit)| {
            (index / classes, index % classes, sigmoid_f32(logit))
        })
        .filter(|(_, _, score)| *score >= options.score_threshold)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.2.total_cmp(&left.2));

    let mut detections = Vec::new();
    for (proposal, label_id, score) in candidates {
        if detections.len() >= options.max_detections {
            break;
        }
        let box_start = proposal * 4;
        let center_x = boxes[box_start] * original_width as f32;
        let center_y = boxes[box_start + 1] * original_height as f32;
        let width = boxes[box_start + 2] * original_width as f32;
        let height = boxes[box_start + 3] * original_height as f32;
        let bbox = [
            (center_x - width * 0.5).clamp(0.0, original_width as f32),
            (center_y - height * 0.5).clamp(0.0, original_height as f32),
            (center_x + width * 0.5).clamp(0.0, original_width as f32),
            (center_y + height * 0.5).clamp(0.0, original_height as f32),
        ];
        let label = labels
            .get(&label_id.to_string())
            .cloned()
            .unwrap_or_else(|| label_id.to_string());
        detections.push(LayoutDetection {
            label,
            score,
            bbox,
            order: i64::try_from(orders[proposal])?,
        });
    }

    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut kept: Vec<LayoutDetection> = Vec::new();
    for detection in detections {
        if kept.len() >= options.max_detections {
            break;
        }
        if options.nms_threshold.is_some_and(|nms_threshold| {
            kept.iter().any(|existing| {
                bbox_iou(existing.bbox, detection.bbox) > nms_threshold
            })
        }) {
            continue;
        }
        kept.push(detection);
    }

    Ok(sort_detections_by_order(kept))
}

fn reading_order_ranks<B: Backend<FloatElem = f32>>(
    order_features: Tensor<B, 3>,
) -> Result<Vec<usize>> {
    let [batch, proposals, features] = order_features.dims();
    if batch != 1 || features % 2 != 0 {
        bail!("PP-DocLayoutV3 order feature shape mismatch");
    }
    let half = features / 2;
    let values = order_features.into_data().to_vec::<f32>()?;
    let mut edges = vec![vec![f32::NEG_INFINITY; proposals]; proposals];
    let mut incoming = vec![0.0; proposals];
    let mut outgoing = vec![0.0; proposals];

    for from in 0..proposals {
        let from_start = from * features;
        for to in 0..proposals {
            if from == to {
                continue;
            }
            let to_start = to * features + half;
            let score = (0..half)
                .map(|index| {
                    values[from_start + index] * values[to_start + index]
                })
                .sum::<f32>();
            edges[from][to] = score;
            outgoing[from] += score;
            incoming[to] += score;
        }
    }

    let mut visited = vec![false; proposals];
    let mut ranks = vec![proposals; proposals];
    let mut current = (0..proposals)
        .min_by(|left, right| {
            (incoming[*left] - outgoing[*left])
                .total_cmp(&(incoming[*right] - outgoing[*right]))
        })
        .unwrap_or(0);

    for rank in 0..proposals {
        ranks[current] = rank;
        visited[current] = true;
        let next = edges[current]
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, _score)| !visited[*index])
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(index, _score)| index)
            .or_else(|| (0..proposals).find(|index| !visited[*index]));
        let Some(next) = next else {
            break;
        };
        current = next;
    }

    Ok(ranks)
}

fn read_f32_tensor<B>(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 1>>
where
    B: Backend<FloatElem = f32>,
{
    Ok(Tensor::from_data(
        TensorData::new(read_f32_values(tensors, name, shape)?, shape.to_vec()),
        device,
    ))
}

fn read_conv1x1_weight<B>(
    tensors: &SafeTensors<'_>,
    name: &str,
    output_channels: usize,
    input_channels: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>>
where
    B: Backend<FloatElem = f32>,
{
    let values = read_f32_values(
        tensors,
        name,
        &[output_channels, input_channels, 1, 1],
    )?;
    let mut transposed = vec![0.0; values.len()];
    for output in 0..output_channels {
        for input in 0..input_channels {
            transposed[input * output_channels + output] =
                values[output * input_channels + input];
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(transposed, [input_channels, output_channels]),
        device,
    ))
}

fn read_conv2d_weight<B>(
    tensors: &SafeTensors<'_>,
    name: &str,
    output_channels: usize,
    input_channels: usize,
    kernel_size: [usize; 2],
    device: &B::Device,
) -> Result<Tensor<B, 4>>
where
    B: Backend<FloatElem = f32>,
{
    Ok(Tensor::from_data(
        TensorData::new(
            read_f32_values(
                tensors,
                name,
                &[
                    output_channels,
                    input_channels,
                    kernel_size[0],
                    kernel_size[1],
                ],
            )?,
            [
                output_channels,
                input_channels,
                kernel_size[0],
                kernel_size[1],
            ],
        ),
        device,
    ))
}

fn read_linear_weight<B>(
    tensors: &SafeTensors<'_>,
    name: &str,
    input_features: usize,
    output_features: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>>
where
    B: Backend<FloatElem = f32>,
{
    let values =
        read_f32_values(tensors, name, &[output_features, input_features])?;
    let mut transposed = vec![0.0; values.len()];
    for output in 0..output_features {
        for input in 0..input_features {
            transposed[input * output_features + output] =
                values[output * input_features + input];
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(transposed, [input_features, output_features]),
        device,
    ))
}

fn flatten_feature_maps<B: Backend<FloatElem = f32>>(
    features: Vec<Tensor<B, 4>>,
) -> Tensor<B, 3> {
    let flattened = features
        .into_iter()
        .map(|feature| {
            let [batch, channels, height, width] = feature.dims();
            feature.swap_dims(1, 3).swap_dims(1, 2).reshape([
                batch,
                height * width,
                channels,
            ])
        })
        .collect();

    Tensor::cat(flattened, 1)
}

fn aifi_position_embedding<B: Backend<FloatElem = f32>>(
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let position_dim = 64usize;
    let temperature = 10_000.0_f32;
    let mut values = Vec::with_capacity(height * width * 256);

    for y in 0..height {
        for x in 0..width {
            for index in 0..position_dim {
                let omega =
                    1.0 / temperature.powf(index as f32 / position_dim as f32);
                values.push((y as f32 * omega).sin());
            }
            for index in 0..position_dim {
                let omega =
                    1.0 / temperature.powf(index as f32 / position_dim as f32);
                values.push((y as f32 * omega).cos());
            }
            for index in 0..position_dim {
                let omega =
                    1.0 / temperature.powf(index as f32 / position_dim as f32);
                values.push((x as f32 * omega).sin());
            }
            for index in 0..position_dim {
                let omega =
                    1.0 / temperature.powf(index as f32 / position_dim as f32);
                values.push((x as f32 * omega).cos());
            }
        }
    }

    Tensor::from_data(TensorData::new(values, [1, height * width, 256]), device)
}

fn topk_proposal_indices<B: Backend<FloatElem = f32>>(
    scores: Tensor<B, 3>,
    k: usize,
) -> Result<Vec<usize>> {
    let [batch, proposals, classes] = scores.dims();
    if batch != 1 {
        bail!("PP-DocLayoutV3 top-k proposal gather expects batch size 1");
    }
    let values = scores.into_data().to_vec::<f32>()?;
    let mut ranked = (0..proposals)
        .map(|proposal| {
            let start = proposal * classes;
            let score = values[start..start + classes]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            (proposal, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));

    Ok(ranked
        .into_iter()
        .take(k)
        .map(|(proposal, _score)| proposal)
        .collect())
}

fn gather_sequence<B: Backend<FloatElem = f32>>(
    tensor: Tensor<B, 3>,
    indices: &[usize],
    width: usize,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, actual_width] = tensor.dims();
    if batch != 1 || actual_width != width {
        bail!("PP-DocLayoutV3 sequence gather shape mismatch");
    }
    let device = tensor.device();
    let values = tensor.into_data().to_vec::<f32>()?;
    let mut gathered = Vec::with_capacity(indices.len() * width);
    for &index in indices {
        if index >= sequence {
            bail!(
                "PP-DocLayoutV3 proposal index {index} out of range {sequence}"
            );
        }
        let start = index * width;
        gathered.extend_from_slice(&values[start..start + width]);
    }

    Ok(Tensor::from_data(
        TensorData::new(gathered, [1, indices.len(), width]),
        &device,
    ))
}

fn deformable_attention_context<B: Backend<FloatElem = f32>>(
    value: Tensor<B, 4>,
    offsets: Tensor<B, 6>,
    weights: Tensor<B, 4>,
    reference_boxes: Tensor<B, 3>,
    spatial_shapes: &[(usize, usize)],
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, heads, head_dim] = value.dims();
    let [
        offset_batch,
        queries,
        offset_heads,
        levels,
        points,
        offset_dims,
    ] = offsets.dims();
    let [weight_batch, weight_queries, weight_heads, weight_points] =
        weights.dims();
    let [reference_batch, reference_queries, reference_dims] =
        reference_boxes.dims();
    if batch != 1
        || offset_batch != 1
        || weight_batch != 1
        || reference_batch != 1
        || offset_heads != heads
        || weight_heads != heads
        || weight_queries != queries
        || reference_queries != queries
        || levels != spatial_shapes.len()
        || points != 4
        || offset_dims != 2
        || weight_points != levels * points
        || reference_dims != 4
    {
        bail!("PP-DocLayoutV3 deformable attention shape mismatch");
    }

    let device = value.device();
    let value_values = value.into_data().to_vec::<f32>()?;
    let offset_values = offsets.into_data().to_vec::<f32>()?;
    let weight_logits = weights.into_data().to_vec::<f32>()?;
    let reference_values = reference_boxes.into_data().to_vec::<f32>()?;
    let mut level_starts = Vec::with_capacity(levels);
    let mut start = 0usize;
    for &(height, width) in spatial_shapes {
        level_starts.push(start);
        start += height * width;
    }
    if start != sequence {
        bail!("PP-DocLayoutV3 deformable attention sequence mismatch");
    }

    let mut context = vec![0.0; queries * heads * head_dim];
    for query in 0..queries {
        let reference_start = query * 4;
        let reference_x = reference_values[reference_start];
        let reference_y = reference_values[reference_start + 1];
        let reference_w = reference_values[reference_start + 2];
        let reference_h = reference_values[reference_start + 3];
        for head in 0..heads {
            let weight_base = (query * heads + head) * levels * points;
            let max_logit = weight_logits
                [weight_base..weight_base + levels * points]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut weight_sum = 0.0f32;
            let mut normalized_weights = [0.0f32; 12];
            for index in 0..levels * points {
                let weight =
                    (weight_logits[weight_base + index] - max_logit).exp();
                normalized_weights[index] = weight;
                weight_sum += weight;
            }
            for weight in normalized_weights.iter_mut().take(levels * points) {
                *weight /= weight_sum;
            }

            for level in 0..levels {
                let (height, width) = spatial_shapes[level];
                for point in 0..points {
                    let offset_start =
                        (((query * heads + head) * levels + level) * points
                            + point)
                            * 2;
                    let x = reference_x
                        + offset_values[offset_start] / points as f32
                            * reference_w
                            * 0.5;
                    let y = reference_y
                        + offset_values[offset_start + 1] / points as f32
                            * reference_h
                            * 0.5;
                    let sample_x = x * width as f32 - 0.5;
                    let sample_y = y * height as f32 - 0.5;
                    let x0 = sample_x.floor() as isize;
                    let y0 = sample_y.floor() as isize;
                    let dx = sample_x - x0 as f32;
                    let dy = sample_y - y0 as f32;
                    let attention_weight =
                        normalized_weights[level * points + point];
                    for dim in 0..head_dim {
                        let mut sampled = 0.0f32;
                        for (iy, wy) in [(y0, 1.0 - dy), (y0 + 1, dy)] {
                            if iy < 0 || iy >= height as isize {
                                continue;
                            }
                            for (ix, wx) in [(x0, 1.0 - dx), (x0 + 1, dx)] {
                                if ix < 0 || ix >= width as isize {
                                    continue;
                                }
                                let sequence_index = level_starts[level]
                                    + iy as usize * width
                                    + ix as usize;
                                let value_index =
                                    (sequence_index * heads + head) * head_dim
                                        + dim;
                                sampled += value_values[value_index] * wx * wy;
                            }
                        }
                        let context_index =
                            (query * heads + head) * head_dim + dim;
                        context[context_index] += sampled * attention_weight;
                    }
                }
            }
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(context, [1, queries, heads * head_dim]),
        &device,
    ))
}

fn generate_encoder_anchors<B: Backend<FloatElem = f32>>(
    features: &[Tensor<B, 4>],
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
    let mut values = Vec::new();
    let mut mask = Vec::new();
    let grid_size = 0.05_f32;
    let eps = 1.0e-2_f32;
    for (level, feature) in features.iter().enumerate() {
        let [_batch, _channels, height, width] = feature.dims();
        let wh = grid_size * 2_f32.powi(i32::try_from(level)?);
        for y in 0..height {
            for x in 0..width {
                let center_x = (x as f32 + 0.5) / width as f32;
                let center_y = (y as f32 + 0.5) / height as f32;
                let valid = center_x > eps
                    && center_x < 1.0 - eps
                    && center_y > eps
                    && center_y < 1.0 - eps
                    && wh > eps
                    && wh < 1.0 - eps;
                let value = if valid { 1.0 } else { 0.0 };
                mask.push(value);
                values.push(if valid {
                    logit(center_x)
                } else {
                    f32::INFINITY
                });
                values.push(if valid {
                    logit(center_y)
                } else {
                    f32::INFINITY
                });
                values.push(if valid { logit(wh) } else { f32::INFINITY });
                values.push(if valid { logit(wh) } else { f32::INFINITY });
            }
        }
    }
    let count = values.len() / 4;

    Ok((
        Tensor::from_data(TensorData::new(values, [1, count, 4]), device),
        Tensor::from_data(TensorData::new(mask, [1, count, 1]), device),
    ))
}

fn logit(value: f32) -> f32 {
    (value / (1.0 - value)).ln()
}

fn read_f32_values(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing PP-DocLayoutV3 tensor {name}"))?;
    if tensor.dtype() != Dtype::F32 {
        bail!("PP-DocLayoutV3 tensor {name} must be float32");
    }
    if tensor.shape() != shape {
        bail!(
            "PP-DocLayoutV3 tensor {name} shape {:?} does not match {:?}",
            tensor.shape(),
            shape
        );
    }

    Ok(tensor
        .data()
        .chunks_exact(4)
        .map(|bytes| {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        })
        .collect())
}

fn inverse_sigmoid_tensor<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    let x = x.clamp(1e-5, 1.0 - 1e-5);
    (x.clone() / (x * -1.0 + 1.0)).log()
}

pub(crate) fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let x1 = left[0].max(right[0]);
    let y1 = left[1].max(right[1]);
    let x2 = left[2].min(right[2]);
    let y2 = left[3].min(right[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area =
        (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}
