use std::collections::HashMap;
use std::sync::LazyLock;

use burn::tensor::{
    Tensor, TensorData, activation::softmax, backend::Backend, module::conv1d,
    ops::ConvOptions,
};
use burn_dispatch::DispatchDevice;
use safetensors::{Dtype, SafeTensors};

use crate::detection::{
    DetectionError, FileType,
    models::magika_preprocess::{PreparedInput, prepare_input},
    vendor::{content::ContentType, file::TypeInfo, model as vendor_model},
};
use crate::ml::safe_matmul;

#[cfg(test)]
mod tests;

const NUM_CLASSES: usize = 257;
const SEQ_LEN: usize = 2048;
const EMBED_DIM: usize = 64;
const TOKENS_PER_BLOCK: usize = 512;
const CHANNELS_PER_TOKEN: usize = 256;
const CONV_OUT_CHANNELS: usize = 512;
const CONV_KERNEL: usize = 5;
const DENSE_OUT: usize = vendor_model::NUM_LABELS;
// Written to `OUT_DIR` by `build.rs`, which converts the committed upstream
// `model.onnx` into safetensors at build time. Embedded directly; the derived
// safetensors is never committed and nothing is fetched at build or runtime.
const EMBEDDED_MODEL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/magika.safetensors"));

// File types missing from the bundled Magika model, reported by it as the
// map key. The corrected metadata replaces the reported type before results
// leave the model.
static MISDETECTIONS: LazyLock<HashMap<ContentType, &TypeInfo>> =
    LazyLock::new(|| {
        HashMap::from([(
            // HEIC shares the ISO-BMFF container with MP4, so Magika reports
            // `mp4` for HEIC photos.
            ContentType::Mp4,
            &TypeInfo {
                label: "heic",
                mime_type: "image/heic",
                group: "image",
                description: "High Efficiency Image File Format",
                extensions: &["heic", "heif"],
                is_text: false,
            },
        )])
    });

/// Returns the corrected metadata for a known Magika misdetection.
///
/// ISO-BMFF is a shared container, so the override only applies when the
/// bytes carry a HEIF brand and genuine MP4 files stay untouched.
fn misdetection_override(
    content_type: ContentType,
    bytes: &[u8],
) -> Option<&'static TypeInfo> {
    let info = MISDETECTIONS.get(&content_type).copied()?;
    if !crate::heif::has_heif_brand(bytes) {
        return None;
    }
    Some(info)
}

struct TensorSpec {
    name: &'static str,
    shape: &'static [usize; 4],
    rank: usize,
}

const EMBEDDING_WEIGHT: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/Const:0",
    shape: &[NUM_CLASSES, EMBED_DIM, 0, 0],
    rank: 2,
};

const EMBEDDING_BIAS: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Dense_0/Reshape:0",
    shape: &[1, 1, EMBED_DIM, 0],
    rank: 3,
};

const LAYER_NORM_0_WEIGHT: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_0/Reshape_2:0",
    shape: &[1, TOKENS_PER_BLOCK, 1, 0],
    rank: 3,
};

const LAYER_NORM_0_BIAS: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_0/Reshape_3:0",
    shape: &[1, TOKENS_PER_BLOCK, 1, 0],
    rank: 3,
};

const CONV_WEIGHT: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Conv_0/transpose_3:0",
    shape: &[CONV_OUT_CHANNELS, CHANNELS_PER_TOKEN, CONV_KERNEL, 1],
    rank: 4,
};

const CONV_BIAS: TensorSpec = TensorSpec {
    name: "const_fold_opt__209",
    shape: &[1, CONV_OUT_CHANNELS, 1, 0],
    rank: 3,
};

const LAYER_NORM_1_WEIGHT: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_1/Reshape_2:0",
    shape: &[1, CONV_OUT_CHANNELS, 0, 0],
    rank: 2,
};

const LAYER_NORM_1_BIAS: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/LayerNorm_1/Reshape_3:0",
    shape: &[1, CONV_OUT_CHANNELS, 0, 0],
    rank: 2,
};

const DENSE_WEIGHT: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/Const_24:0",
    shape: &[CONV_OUT_CHANNELS, DENSE_OUT, 0, 0],
    rank: 2,
};

const DENSE_BIAS: TensorSpec = TensorSpec {
    name: "jax2tf_get_logits_/pjit_get_logits_/MagikaV2/Dense_1/Reshape:0",
    shape: &[1, DENSE_OUT, 0, 0],
    rank: 2,
};

/// The Magika file-type classifier.
pub struct MagikaModel<B: Backend> {
    device: B::Device,
    embedding_table: Vec<f32>,
    layer_norm_0_weight: Tensor<B, 3>,
    layer_norm_0_bias: Tensor<B, 3>,
    conv_weight: Tensor<B, 3>,
    conv_bias: Tensor<B, 1>,
    layer_norm_1_weight: Tensor<B, 2>,
    layer_norm_1_bias: Tensor<B, 2>,
    dense_weight: Tensor<B, 2>,
    dense_bias: Tensor<B, 2>,
}

/// Per-input classification result.
enum RowOutcome {
    Ruled(ContentType),
    Scored(usize, f32),
}

impl<B: Backend<FloatElem = f32, Device = DispatchDevice>> MagikaModel<B> {
    /// Loads the model from its bundled weights.
    pub fn from_embedded(device: &B::Device) -> Result<Self, DetectionError> {
        Self::from_bytes(device, EMBEDDED_MODEL)
    }

    /// Loads a model from raw weight bytes.
    pub fn from_bytes(
        device: &B::Device,
        model_bytes: &[u8],
    ) -> Result<Self, DetectionError> {
        let initializers =
            SafeTensors::deserialize(model_bytes).map_err(|source| {
                DetectionError::Model {
                    operation: "parse weights",
                    source: Box::new(source),
                }
            })?;

        let embedding_weight =
            read_tensor_spec(&initializers, &EMBEDDING_WEIGHT)?;
        let embedding_bias = read_tensor_spec(&initializers, &EMBEDDING_BIAS)?;
        // Preserve host bias addition and backend GELU, once per token at load.
        let embedding_table = gelu(tensor_2d_from_flat::<B>(
            device,
            embedding_weight
                .into_iter()
                .enumerate()
                .map(|(index, weight)| {
                    weight + embedding_bias[index % EMBED_DIM]
                })
                .collect(),
            [NUM_CLASSES, EMBED_DIM],
        ))
        .into_data()
        .to_vec::<f32>()
        .map_err(|source| DetectionError::Model {
            operation: "read embedding table",
            source: Box::new(source),
        })?;

        Ok(Self {
            device: (*device).clone(),
            embedding_table,
            layer_norm_0_weight: tensor_3d(
                device,
                &initializers,
                &LAYER_NORM_0_WEIGHT,
                [1, TOKENS_PER_BLOCK, 1],
            )?,
            layer_norm_0_bias: tensor_3d(
                device,
                &initializers,
                &LAYER_NORM_0_BIAS,
                [1, TOKENS_PER_BLOCK, 1],
            )?,
            conv_weight: tensor_3d(
                device,
                &initializers,
                &CONV_WEIGHT,
                [CONV_OUT_CHANNELS, CHANNELS_PER_TOKEN, CONV_KERNEL],
            )?,
            conv_bias: tensor_1d_from_flat(
                device,
                read_tensor_spec(&initializers, &CONV_BIAS)?,
            ),
            layer_norm_1_weight: tensor_2d_from_flat(
                device,
                read_tensor_spec(&initializers, &LAYER_NORM_1_WEIGHT)?,
                [1, CONV_OUT_CHANNELS],
            ),
            layer_norm_1_bias: tensor_2d_from_flat(
                device,
                read_tensor_spec(&initializers, &LAYER_NORM_1_BIAS)?,
                [1, CONV_OUT_CHANNELS],
            ),
            dense_weight: tensor_2d_from_flat(
                device,
                read_tensor_spec(&initializers, &DENSE_WEIGHT)?,
                [CONV_OUT_CHANNELS, DENSE_OUT],
            ),
            dense_bias: tensor_2d_from_flat(
                device,
                read_tensor_spec(&initializers, &DENSE_BIAS)?,
                [1, DENSE_OUT],
            ),
        })
    }

    /// Resolves a single [`FileType`] for raw bytes.
    pub fn identify_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<FileType, DetectionError> {
        let mut all = self.detect_content_type_batch(vec![bytes])?;
        Ok(all.remove(0))
    }

    fn detect_content_type_batch(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<FileType>, DetectionError> {
        self.classify(inputs.clone())?
            .into_iter()
            .zip(inputs)
            .map(|(outcome, bytes)| match outcome {
                RowOutcome::Ruled(content_type) => {
                    Ok(match misdetection_override(content_type, bytes) {
                        Some(info) => FileType::ruled_info(info),
                        None => FileType::ruled(content_type),
                    })
                }
                RowOutcome::Scored(label_idx, score) => {
                    let content_type =
                        self.final_content_type(label_idx, score)?;
                    Ok(match misdetection_override(content_type, bytes) {
                        Some(info) => FileType::ruled_info(info),
                        None => FileType::inferred(content_type, score),
                    })
                }
            })
            .collect()
    }

    /// Classifies each input, returning one [`RowOutcome`] in input order.
    fn classify(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<RowOutcome>, DetectionError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut outcomes: Vec<Option<RowOutcome>> =
            (0..inputs.len()).map(|_| None).collect();
        let mut pending_positions = Vec::new();
        let mut pending_features = Vec::new();

        for (index, bytes) in inputs.into_iter().enumerate() {
            match prepare_input(bytes, &vendor_model::CONFIG) {
                PreparedInput::Ruled(content_type) => {
                    outcomes[index] = Some(RowOutcome::Ruled(content_type));
                }
                PreparedInput::Features(features) => {
                    pending_positions.push(index);
                    pending_features.push(features);
                }
            }
        }

        if !pending_features.is_empty() {
            let scores = self
                .forward(&pending_features)?
                .into_data()
                .to_vec::<f32>()
                .map_err(|source| DetectionError::Model {
                    operation: "read tensor output",
                    source: Box::new(source),
                })?;
            if scores.len() != pending_positions.len() * DENSE_OUT {
                return Err(DetectionError::InvalidModel {
                    message: "runtime returned mismatched batch size"
                        .to_owned(),
                });
            }

            for (position, row) in pending_positions
                .into_iter()
                .zip(scores.chunks_exact(DENSE_OUT))
            {
                let (label_idx, score) = best_score(row)?;
                outcomes[position] = Some(RowOutcome::Scored(label_idx, score));
            }
        }

        outcomes
            .into_iter()
            .map(|outcome| {
                outcome.ok_or_else(|| DetectionError::InvalidModel {
                    message: "missing detection result".to_owned(),
                })
            })
            .collect()
    }

    fn forward(
        &self,
        batch_features: &[Vec<i32>],
    ) -> Result<Tensor<B, 2>, DetectionError> {
        let batch_size = batch_features.len();
        let feature_count = batch_features.iter().map(Vec::len).sum::<usize>();

        if feature_count != batch_size * SEQ_LEN {
            return Err(DetectionError::InvalidModel {
                message: "unexpected feature batch shape".to_owned(),
            });
        }

        let mut embedded = Vec::with_capacity(batch_size * SEQ_LEN * EMBED_DIM);
        for features in batch_features {
            for &feature in features {
                let index = usize::try_from(feature).map_err(|_| {
                    DetectionError::InvalidModel {
                        message: format!("negative feature value: {feature}"),
                    }
                })?;
                if index >= NUM_CLASSES {
                    return Err(DetectionError::InvalidModel {
                        message: format!(
                            "feature value out of range: {feature}"
                        ),
                    });
                }

                let start = index * EMBED_DIM;
                embedded.extend_from_slice(
                    &self.embedding_table[start..start + EMBED_DIM],
                );
            }
        }

        let x = Tensor::<B, 3>::from_data(
            TensorData::new(embedded, [batch_size, SEQ_LEN, EMBED_DIM]),
            &self.device,
        );
        let x: Tensor<B, 3> =
            x.reshape([batch_size, TOKENS_PER_BLOCK, CHANNELS_PER_TOKEN]);
        let x = layer_norm_axis_1_3d(
            x,
            TOKENS_PER_BLOCK as f32,
            self.layer_norm_0_weight.clone(),
            self.layer_norm_0_bias.clone(),
        );
        let x = x.permute([0, 2, 1]);
        let x = self.convolve(x);
        let x = gelu(x);
        let pooled = x.max_dim(2).squeeze_dim(2);

        let normalized = layer_norm_axis_1_2d(
            pooled,
            CONV_OUT_CHANNELS as f32,
            self.layer_norm_1_weight.clone(),
            self.layer_norm_1_bias.clone(),
        );
        let logits = normalized.matmul(self.dense_weight.clone())
            + self.dense_bias.clone();
        Ok(softmax(logits, 1))
    }

    fn convolve(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Native convolution is faster on CPU; the five GEMMs benefit WGPU.
        if let DispatchDevice::Flex(_) = &self.device {
            return conv1d(
                x,
                self.conv_weight.clone(),
                Some(self.conv_bias.clone()),
                ConvOptions::new([1], [0], [1], 1),
            );
        }
        let output_len = TOKENS_PER_BLOCK - CONV_KERNEL + 1;
        let mut acc: Option<Tensor<B, 3>> = None;
        for k in 0..CONV_KERNEL {
            let weight = self.conv_weight.clone().narrow(2, k, 1).reshape([
                1,
                CONV_OUT_CHANNELS,
                CHANNELS_PER_TOKEN,
            ]);
            let part = safe_matmul(weight, x.clone().narrow(2, k, output_len));
            acc = Some(match acc {
                None => part,
                Some(previous) => previous + part,
            });
        }
        acc.expect("CONV_KERNEL is five, so the accumulator is populated")
            + self.conv_bias.clone().reshape([1, CONV_OUT_CHANNELS, 1])
    }

    fn final_content_type(
        &self,
        label_idx: usize,
        score: f32,
    ) -> Result<ContentType, DetectionError> {
        let inferred_type = label_for_index(label_idx)?.content_type();
        if score < vendor_model::CONFIG.thresholds[inferred_type as usize] {
            return Ok(if inferred_type.info().is_text {
                ContentType::Txt
            } else {
                ContentType::Unknown
            });
        }

        Ok(vendor_model::CONFIG.overwrite_map[inferred_type as usize])
    }
}

/// Returns a probability row's highest-scoring label.
pub(in crate::detection) fn best_score(
    row: &[f32],
) -> Result<(usize, f32), DetectionError> {
    if row.len() != DENSE_OUT {
        return Err(DetectionError::InvalidModel {
            message: format!("unexpected probability row size: {}", row.len()),
        });
    }
    let mut best = (0, f32::NEG_INFINITY);
    for (index, &score) in row.iter().enumerate() {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(DetectionError::InvalidModel {
                message: "probability row contains an invalid score".to_owned(),
            });
        }
        if score.total_cmp(&best.1).is_gt() {
            best = (index, score);
        }
    }

    Ok(best)
}

fn tensor_2d_from_flat<B: Backend<FloatElem = f32>>(
    device: &B::Device,
    values: Vec<f32>,
    shape: [usize; 2],
) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(values, shape), device)
}

fn tensor_1d_from_flat<B: Backend<FloatElem = f32>>(
    device: &B::Device,
    values: Vec<f32>,
) -> Tensor<B, 1> {
    let len = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values, [len]), device)
}

fn tensor_3d<B: Backend<FloatElem = f32>>(
    device: &B::Device,
    initializers: &SafeTensors<'_>,
    spec: &TensorSpec,
    shape: [usize; 3],
) -> Result<Tensor<B, 3>, DetectionError> {
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(read_tensor_spec(initializers, spec)?, shape),
        device,
    ))
}

fn read_tensor_spec(
    initializers: &SafeTensors<'_>,
    spec: &TensorSpec,
) -> Result<Vec<f32>, DetectionError> {
    read_f32_tensor(initializers, spec.name, &spec.shape[..spec.rank])
}

fn read_f32_tensor(
    initializers: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, DetectionError> {
    let tensor =
        initializers
            .tensor(name)
            .map_err(|source| DetectionError::Model {
                operation: "read weight",
                source: Box::new(source),
            })?;

    if tensor.dtype() != Dtype::F32 {
        return Err(DetectionError::InvalidModel {
            message: format!(
                "weight {name} has unexpected dtype {:?}",
                tensor.dtype()
            ),
        });
    }

    if tensor.shape() != expected_shape {
        return Err(DetectionError::InvalidModel {
            message: format!(
                "weight {name} has shape {:?}, expected {:?}",
                tensor.shape(),
                expected_shape
            ),
        });
    }

    let values = tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect::<Vec<_>>();

    if values.len() != expected_shape.iter().product::<usize>() {
        return Err(DetectionError::InvalidModel {
            message: format!(
                "weight {name} has {} values, expected {}",
                values.len(),
                expected_shape.iter().product::<usize>()
            ),
        });
    }

    Ok(values)
}

fn gelu<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    let cubic = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cubic * 0.044_715) * 0.797_884_6;
    x * ((inner.tanh() + 1.0) * 0.5)
}

fn layer_norm_axis_1_3d<
    B: Backend<FloatElem = f32, Device = DispatchDevice>,
>(
    x: Tensor<B, 3>,
    axis_len: f32,
    weight: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let batch = x.dims()[0];
    if batch > 1 && matches!(x.device(), DispatchDevice::Flex(_)) {
        // SIMD norm0 measured ~0.079 ms at B1 vs ~91 ms at B32: avoid batched broadcasts.
        return Tensor::cat(
            (0..batch)
                .map(|index| {
                    layer_norm_axis_1_3d(
                        x.clone().narrow(0, index, 1),
                        axis_len,
                        weight.clone(),
                        bias.clone(),
                    )
                })
                .collect(),
            0,
        );
    }
    let mean = x.clone().sum_dim(1) * (1.0 / axis_len);
    let variance = (x.clone() * x.clone()).sum_dim(1) * (1.0 / axis_len)
        - mean.clone() * mean.clone();
    let inv_std = (variance.clamp_min(0.0) + 1e-6).sqrt().recip();
    ((x - mean) * inv_std) * weight + bias
}

fn layer_norm_axis_1_2d<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 2>,
    axis_len: f32,
    weight: Tensor<B, 2>,
    bias: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let mean = x.clone().sum_dim(1) * (1.0 / axis_len);
    let variance = (x.clone() * x.clone()).sum_dim(1) * (1.0 / axis_len)
        - mean.clone() * mean.clone();
    let inv_std = (variance.clamp_min(0.0) + 1e-6).sqrt().recip();
    ((x - mean) * inv_std) * weight + bias
}

fn label_for_index(
    index: usize,
) -> Result<vendor_model::Label, DetectionError> {
    if index >= vendor_model::NUM_LABELS {
        return Err(DetectionError::InvalidModel {
            message: format!("label index out of range: {index}"),
        });
    }

    Ok(
        // SAFETY: `index < NUM_LABELS` checked above; `Label` is `#[repr(u32)]`
        // with exactly `NUM_LABELS` variants, so the transmute is in-range.
        unsafe {
            std::mem::transmute::<u32, vendor_model::Label>(index as u32)
        },
    )
}
