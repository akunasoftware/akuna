use std::fmt;

use burn::tensor::{
    Tensor, TensorData, activation::softmax, backend::Backend, module::conv1d,
    ops::ConvOptions,
};
use safetensors::{Dtype, SafeTensors};

#[cfg(test)]
use crate::detection::{Detection, RankedAlternative};
use crate::detection::{
    FileType,
    models::magika_preprocess::{PreparedInput, prepare_input},
    vendor::{content::ContentType, model as vendor_model},
};

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

/// Errors raised while loading or running the Magika model.
#[derive(Debug)]
pub enum MagikaInferenceError {
    /// An I/O failure.
    Io(std::io::Error),
    /// Invalid or incompatible model configuration.
    InvalidConfig(String),
    /// A failure during inference or weight loading.
    Runtime(String),
}

impl fmt::Display for MagikaInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::InvalidConfig(e) => write!(f, "invalid configuration: {e}"),
            Self::Runtime(e) => write!(f, "inference runtime error: {e}"),
        }
    }
}

impl std::error::Error for MagikaInferenceError {}

impl From<std::io::Error> for MagikaInferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// The Magika file-type classifier.
pub struct MagikaModel<B: Backend> {
    device: B::Device,
    #[cfg(test)]
    top_k: usize,
    embedding_weight: Vec<f32>,
    embedding_bias: Vec<f32>,
    layer_norm_0_weight: Tensor<B, 3>,
    layer_norm_0_bias: Tensor<B, 3>,
    conv_weight: Tensor<B, 3>,
    conv_bias: Tensor<B, 1>,
    layer_norm_1_weight: Tensor<B, 2>,
    layer_norm_1_bias: Tensor<B, 2>,
    dense_weight: Tensor<B, 2>,
    dense_bias: Tensor<B, 2>,
}

/// Per-input classification result: either a rule-resolved type or a scored
/// list of `(label_idx, score)` pairs.
enum RowOutcome {
    Ruled(ContentType),
    Scored(Vec<(usize, f32)>),
}

impl<B: Backend<FloatElem = f32>> MagikaModel<B> {
    /// Loads the model from its bundled weights.
    pub fn from_embedded(
        device: &B::Device,
    ) -> Result<Self, MagikaInferenceError> {
        Self::from_bytes(device, EMBEDDED_MODEL)
    }

    /// Loads a model from raw weight bytes.
    pub fn from_bytes(
        device: &B::Device,
        model_bytes: &[u8],
    ) -> Result<Self, MagikaInferenceError> {
        let initializers =
            SafeTensors::deserialize(model_bytes).map_err(|err| {
                MagikaInferenceError::Runtime(format!("parse weights: {err}"))
            })?;

        Ok(Self {
            device: (*device).clone(),
            #[cfg(test)]
            top_k: 3,
            embedding_weight: read_tensor_spec(
                &initializers,
                &EMBEDDING_WEIGHT,
            )?,
            embedding_bias: read_tensor_spec(&initializers, &EMBEDDING_BIAS)?,
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
            conv_weight: tensor_3d_from_flat(
                device,
                read_conv_weight(&initializers)?,
                [CONV_OUT_CHANNELS, CHANNELS_PER_TOKEN, CONV_KERNEL],
            ),
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

    /// Overrides the number of ranked alternatives returned per detection.
    #[cfg(test)]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Classifies raw bytes and returns ranked alternatives.
    #[cfg(test)]
    pub fn detect_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<Detection, MagikaInferenceError> {
        let mut all = self.detect_batch(vec![bytes])?;
        Ok(all.remove(0))
    }

    /// Resolves a single [`FileType`] for raw bytes.
    pub fn identify_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        let mut all = self.detect_content_type_batch(vec![bytes])?;
        let content_type = all.remove(0);
        Ok(FileType::Ruled(content_type))
    }

    /// Classifies a batch of inputs, returning ranked alternatives for each.
    #[cfg(test)]
    pub fn detect_batch(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<Detection>, MagikaInferenceError> {
        self.classify(inputs)?
            .into_iter()
            .map(|outcome| match outcome {
                RowOutcome::Ruled(content_type) => {
                    Ok(detection_for_content_type(content_type))
                }
                RowOutcome::Scored(sorted) => {
                    self.detection_from_sorted(sorted)
                }
            })
            .collect()
    }

    fn detect_content_type_batch(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<ContentType>, MagikaInferenceError> {
        self.classify(inputs)?
            .into_iter()
            .map(|outcome| match outcome {
                RowOutcome::Ruled(content_type) => Ok(content_type),
                RowOutcome::Scored(sorted) => {
                    let (label_idx, score) =
                        sorted.first().copied().ok_or_else(|| {
                            MagikaInferenceError::Runtime(
                                "no alternatives created".to_string(),
                            )
                        })?;
                    self.final_content_type(label_idx, score)
                }
            })
            .collect()
    }

    /// Classifies each input, returning one [`RowOutcome`] in input order.
    fn classify(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<RowOutcome>, MagikaInferenceError> {
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
            let rows = self.infer_rows(&pending_features)?;
            if rows.len() != pending_positions.len() {
                return Err(MagikaInferenceError::Runtime(
                    "runtime returned mismatched batch size".to_string(),
                ));
            }

            for (position, row) in pending_positions.into_iter().zip(rows) {
                outcomes[position] = Some(RowOutcome::Scored(sorted_row(row)?));
            }
        }

        outcomes
            .into_iter()
            .map(|outcome| {
                outcome.ok_or_else(|| {
                    MagikaInferenceError::Runtime(
                        "missing detection result".to_string(),
                    )
                })
            })
            .collect()
    }

    /// Builds a [`Detection`] from a sorted `(label_idx, score)` list.
    #[cfg(test)]
    fn detection_from_sorted(
        &self,
        sorted: Vec<(usize, f32)>,
    ) -> Result<Detection, MagikaInferenceError> {
        let alternatives = sorted
            .iter()
            .take(self.top_k)
            .map(|(label_idx, score)| {
                let content_type =
                    self.final_content_type(*label_idx, *score)?;
                Ok(alternative_for_content_type(content_type, *score))
            })
            .collect::<Result<Vec<_>, MagikaInferenceError>>()?;

        let best = alternatives
            .first()
            .ok_or_else(|| {
                MagikaInferenceError::Runtime(
                    "no alternatives created".to_string(),
                )
            })?
            .clone();

        Ok(Detection {
            label: best.label.clone(),
            mime_type: best.mime_type.clone(),
            confidence: best.confidence,
            alternatives,
        })
    }

    fn forward(
        &self,
        batch_features: &[Vec<i32>],
    ) -> Result<Tensor<B, 2>, MagikaInferenceError> {
        let batch_size = batch_features.len();
        let flat = batch_features
            .iter()
            .flat_map(|features| features.iter().map(|value| *value as f32))
            .collect::<Vec<_>>();

        if flat.len() != batch_size * SEQ_LEN {
            return Err(MagikaInferenceError::Runtime(
                "unexpected feature batch shape".to_string(),
            ));
        }

        let mut embedded = Vec::with_capacity(batch_size * SEQ_LEN * EMBED_DIM);
        for features in batch_features {
            for &feature in features {
                let index = usize::try_from(feature).map_err(|_| {
                    MagikaInferenceError::Runtime(format!(
                        "negative feature value: {feature}"
                    ))
                })?;
                if index >= NUM_CLASSES {
                    return Err(MagikaInferenceError::Runtime(format!(
                        "feature value out of range: {feature}"
                    )));
                }

                let start = index * EMBED_DIM;
                for offset in 0..EMBED_DIM {
                    embedded.push(
                        self.embedding_weight[start + offset]
                            + self.embedding_bias[offset],
                    );
                }
            }
        }

        let x = Tensor::<B, 3>::from_data(
            TensorData::new(embedded, [batch_size, SEQ_LEN, EMBED_DIM]),
            &self.device,
        );
        let x = gelu(x);
        let x: Tensor<B, 3> =
            x.reshape([batch_size, TOKENS_PER_BLOCK, CHANNELS_PER_TOKEN]);
        let x = layer_norm_axis_1_3d(
            x,
            TOKENS_PER_BLOCK as f32,
            self.layer_norm_0_weight.clone(),
            self.layer_norm_0_bias.clone(),
        );
        let x = x.permute([0, 2, 1]);
        let x = conv1d(
            x,
            self.conv_weight.clone(),
            Some(self.conv_bias.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
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

    /// Runs the batch once and splits the probabilities into per-input rows.
    fn infer_rows(
        &self,
        batch_features: &[Vec<i32>],
    ) -> Result<Vec<Vec<f32>>, MagikaInferenceError> {
        let probs = self.forward(batch_features)?;
        let flat = probs.into_data().to_vec::<f32>().map_err(|err| {
            MagikaInferenceError::Runtime(format!("extract tensor data: {err}"))
        })?;

        Ok(flat.chunks(DENSE_OUT).map(|chunk| chunk.to_vec()).collect())
    }

    fn final_content_type(
        &self,
        label_idx: usize,
        score: f32,
    ) -> Result<ContentType, MagikaInferenceError> {
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

/// Returns a probability row's entries enumerated and sorted by score descending.
fn sorted_row(
    row: Vec<f32>,
) -> Result<Vec<(usize, f32)>, MagikaInferenceError> {
    if row.len() != DENSE_OUT {
        return Err(MagikaInferenceError::Runtime(format!(
            "unexpected logits row size: {}",
            row.len()
        )));
    }

    let mut indexed: Vec<(usize, f32)> = row.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(indexed)
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

fn read_conv_weight(
    initializers: &SafeTensors<'_>,
) -> Result<Vec<f32>, MagikaInferenceError> {
    let raw = read_tensor_spec(initializers, &CONV_WEIGHT)?;
    let mut flattened = Vec::with_capacity(
        CONV_OUT_CHANNELS * CHANNELS_PER_TOKEN * CONV_KERNEL,
    );

    for out in 0..CONV_OUT_CHANNELS {
        for channel in 0..CHANNELS_PER_TOKEN {
            for kernel in 0..CONV_KERNEL {
                let index =
                    (out * CHANNELS_PER_TOKEN + channel) * CONV_KERNEL + kernel;
                flattened.push(raw[index]);
            }
        }
    }

    Ok(flattened)
}

fn tensor_3d_from_flat<B: Backend<FloatElem = f32>>(
    device: &B::Device,
    values: Vec<f32>,
    shape: [usize; 3],
) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(values, shape), device)
}

fn tensor_3d<B: Backend<FloatElem = f32>>(
    device: &B::Device,
    initializers: &SafeTensors<'_>,
    spec: &TensorSpec,
    shape: [usize; 3],
) -> Result<Tensor<B, 3>, MagikaInferenceError> {
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(read_tensor_spec(initializers, spec)?, shape),
        device,
    ))
}

fn read_tensor_spec(
    initializers: &SafeTensors<'_>,
    spec: &TensorSpec,
) -> Result<Vec<f32>, MagikaInferenceError> {
    read_f32_tensor(initializers, spec.name, &spec.shape[..spec.rank])
}

fn read_f32_tensor(
    initializers: &SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, MagikaInferenceError> {
    let tensor = initializers.tensor(name).map_err(|_| {
        MagikaInferenceError::Runtime(format!("missing weight: {name}"))
    })?;

    if tensor.dtype() != Dtype::F32 {
        return Err(MagikaInferenceError::Runtime(format!(
            "weight {name} has unexpected dtype {:?}",
            tensor.dtype()
        )));
    }

    if tensor.shape() != expected_shape {
        return Err(MagikaInferenceError::Runtime(format!(
            "weight {name} has shape {:?}, expected {:?}",
            tensor.shape(),
            expected_shape
        )));
    }

    let values = tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
        .collect::<Vec<_>>();

    if values.len() != expected_shape.iter().product::<usize>() {
        return Err(MagikaInferenceError::Runtime(format!(
            "weight {name} has {} values, expected {}",
            values.len(),
            expected_shape.iter().product::<usize>()
        )));
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

fn layer_norm_axis_1_3d<B: Backend<FloatElem = f32>>(
    x: Tensor<B, 3>,
    axis_len: f32,
    weight: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
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
) -> Result<vendor_model::Label, MagikaInferenceError> {
    if index >= vendor_model::NUM_LABELS {
        return Err(MagikaInferenceError::Runtime(format!(
            "label index out of range: {index}"
        )));
    }

    Ok(
        // SAFETY: `index < NUM_LABELS` checked above; `Label` is `#[repr(u32)]`
        // with exactly `NUM_LABELS` variants, so the transmute is in-range.
        unsafe {
            std::mem::transmute::<u32, vendor_model::Label>(index as u32)
        },
    )
}

#[cfg(test)]
fn detection_for_content_type(content_type: ContentType) -> Detection {
    let alternative = alternative_for_content_type(content_type, 1.0);

    Detection {
        label: alternative.label.clone(),
        mime_type: alternative.mime_type.clone(),
        confidence: alternative.confidence,
        alternatives: vec![alternative],
    }
}

#[cfg(test)]
fn alternative_for_content_type(
    content_type: ContentType,
    confidence: f32,
) -> RankedAlternative {
    let info = content_type.info();

    RankedAlternative {
        label: info.label.to_string(),
        mime_type: Some(info.mime_type.to_string()),
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use crate::ml::backend::{Backend, cpu_device};

    use super::MagikaModel;

    #[test]
    fn classifier_batch_is_deterministic() {
        let classifier = MagikaModel::<Backend>::from_embedded(&cpu_device())
            .expect("build classifier");

        let a = classifier
            .detect_bytes(b"abcdef")
            .expect("first inference should succeed");
        let b = classifier
            .detect_bytes(b"abcdef")
            .expect("second inference should succeed");
        assert_eq!(a, b);

        let batch = classifier
            .detect_batch(vec![b"a", b"b", b"c"])
            .expect("batch inference should succeed");
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn embedded_model_builds() {
        MagikaModel::<Backend>::from_embedded(&cpu_device())
            .expect("build embedded model");
    }

    #[test]
    fn explicit_top_k_is_applied() {
        let classifier = MagikaModel::<Backend>::from_embedded(&cpu_device())
            .expect("build model")
            .with_top_k(5);

        let detection = classifier
            .detect_bytes(b"function greet() { return 'hi'; }")
            .expect("detect bytes");
        assert_eq!(detection.alternatives.len(), 5);
    }
}
