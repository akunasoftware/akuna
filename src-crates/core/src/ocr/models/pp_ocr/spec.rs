use crate::ocr::{OcrDetectionModel, OcrRecognitionModel};

/// Returns the weights repo for the given detector.
pub(crate) fn det_safetensors_repo(model: OcrDetectionModel) -> &'static str {
    match model {
        OcrDetectionModel::PpOcrV6Tiny => {
            "PaddlePaddle/PP-OCRv6_tiny_det_safetensors"
        }
        OcrDetectionModel::PpOcrV6Small => {
            "PaddlePaddle/PP-OCRv6_small_det_safetensors"
        }
        OcrDetectionModel::PpOcrV6Medium => {
            "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
        }
    }
}

/// Returns the weights repo for the given recognizer.
pub(crate) fn rec_safetensors_repo(model: OcrRecognitionModel) -> &'static str {
    match model {
        OcrRecognitionModel::PpOcrV6Tiny => {
            "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors"
        }
        OcrRecognitionModel::PpOcrV6Small => {
            "PaddlePaddle/PP-OCRv6_small_rec_safetensors"
        }
        OcrRecognitionModel::PpOcrV6Medium => {
            "PaddlePaddle/PP-OCRv6_medium_rec_safetensors"
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrModelSpec {
    pub(crate) repo_id: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) static_shape: [usize; 4],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrDetectionConfig {
    pub(crate) limit_side_len: u32,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) db_thresh: f32,
    pub(crate) db_box_thresh: f32,
    pub(crate) db_unclip_ratio: f32,
    pub(crate) max_candidates: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrRecognitionConfig {
    pub(crate) spec: PpOcrModelSpec,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) num_classes: usize,
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(crate) fn detector_config(
    _model: OcrDetectionModel,
) -> PpOcrDetectionConfig {
    PpOcrDetectionConfig {
        limit_side_len: 64,
        mean: MEAN,
        std: STD,
        db_thresh: 0.3,
        db_box_thresh: 0.6,
        db_unclip_ratio: 1.5,
        max_candidates: 3000,
    }
}

pub(crate) fn recognizer_config(
    model: OcrRecognitionModel,
) -> PpOcrRecognitionConfig {
    let (repo_id, revision, num_classes) = match model {
        OcrRecognitionModel::PpOcrV6Tiny => (
            "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
            "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
            6906,
        ),
        OcrRecognitionModel::PpOcrV6Small => (
            "PaddlePaddle/PP-OCRv6_small_rec_onnx",
            "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
            18710,
        ),
        OcrRecognitionModel::PpOcrV6Medium => (
            "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
            "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
            18710,
        ),
    };

    PpOcrRecognitionConfig {
        spec: PpOcrModelSpec {
            repo_id,
            revision,
            static_shape: [1, 3, 48, 320],
        },
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        num_classes,
    }
}
