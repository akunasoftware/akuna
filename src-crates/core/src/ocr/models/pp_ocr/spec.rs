use crate::ocr::{OcrDetector, OcrRecognizer};

/// Returns the weights repo for the given detector.
pub(crate) fn det_safetensors_repo(detector: OcrDetector) -> &'static str {
    match detector {
        OcrDetector::PpOcrV6TinyDet => {
            "PaddlePaddle/PP-OCRv6_tiny_det_safetensors"
        }
        OcrDetector::PpOcrV6SmallDet => {
            "PaddlePaddle/PP-OCRv6_small_det_safetensors"
        }
        OcrDetector::PpOcrV6MediumDet => {
            "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
        }
    }
}

/// Returns the weights repo for the given recognizer.
pub(crate) fn rec_safetensors_repo(recognizer: OcrRecognizer) -> &'static str {
    match recognizer {
        OcrRecognizer::PpOcrV6TinyRec => {
            "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors"
        }
        OcrRecognizer::PpOcrV6SmallRec => {
            "PaddlePaddle/PP-OCRv6_small_rec_safetensors"
        }
        OcrRecognizer::PpOcrV6MediumRec => {
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
pub(crate) struct PpOcrDetectorConfig {
    pub(crate) limit_side_len: u32,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) db_thresh: f32,
    pub(crate) db_box_thresh: f32,
    pub(crate) db_unclip_ratio: f32,
    pub(crate) max_candidates: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrRecognizerConfig {
    pub(crate) spec: PpOcrModelSpec,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) num_classes: usize,
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(crate) fn detector_config(detector: OcrDetector) -> PpOcrDetectorConfig {
    let db_box_thresh = match detector {
        OcrDetector::PpOcrV6TinyDet => 0.4,
        OcrDetector::PpOcrV6SmallDet => 0.45,
        OcrDetector::PpOcrV6MediumDet => 0.45,
    };

    PpOcrDetectorConfig {
        limit_side_len: 960,
        mean: MEAN,
        std: STD,
        db_thresh: 0.3,
        db_box_thresh,
        db_unclip_ratio: 1.5,
        max_candidates: 3000,
    }
}

pub(crate) fn recognizer_config(
    recognizer: OcrRecognizer,
) -> PpOcrRecognizerConfig {
    let (repo_id, revision, num_classes) = match recognizer {
        OcrRecognizer::PpOcrV6TinyRec => (
            "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
            "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
            6906,
        ),
        OcrRecognizer::PpOcrV6SmallRec => (
            "PaddlePaddle/PP-OCRv6_small_rec_onnx",
            "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
            18710,
        ),
        OcrRecognizer::PpOcrV6MediumRec => (
            "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
            "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
            18710,
        ),
    };

    PpOcrRecognizerConfig {
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
