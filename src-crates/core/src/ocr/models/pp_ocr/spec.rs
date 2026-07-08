use crate::ml::HfWeight;
use crate::ocr::{OcrDetectionModel, OcrRecognitionModel};

const SAFETENSORS_FILE: &str = "model.safetensors";

/// Returns the pinned weights for the given detector.
pub(crate) fn detector_weight(model: OcrDetectionModel) -> HfWeight {
    match model {
        OcrDetectionModel::PpOcrV6Tiny => HfWeight {
            repo_id: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
            revision: "07595f982703daf0d4e120a12a01da8073542f3a",
            filename: SAFETENSORS_FILE,
        },
        OcrDetectionModel::PpOcrV6Small => HfWeight {
            repo_id: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
            revision: "eae2ee920a39fb3087637d3dbb58df1896ec1f24",
            filename: SAFETENSORS_FILE,
        },
        OcrDetectionModel::PpOcrV6Medium => HfWeight {
            repo_id: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
            revision: "4236c2b61741a259c091fd879dcc4edc339e916c",
            filename: SAFETENSORS_FILE,
        },
    }
}

/// Returns the pinned weights for the given recognizer.
pub(crate) fn recognizer_weight(model: OcrRecognitionModel) -> HfWeight {
    match model {
        OcrRecognitionModel::PpOcrV6Tiny => HfWeight {
            repo_id: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
            revision: "6f2d2d51b4b4226d7a2329a02f416f4994106f3a",
            filename: SAFETENSORS_FILE,
        },
        OcrRecognitionModel::PpOcrV6Small => HfWeight {
            repo_id: "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
            revision: "fe049fb103f57443fe8840c54ed06b702f3c1de5",
            filename: SAFETENSORS_FILE,
        },
        OcrRecognitionModel::PpOcrV6Medium => HfWeight {
            repo_id: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
            revision: "024cad6a831de75c2c3c26e711ba8c4a82ccd24b",
            filename: SAFETENSORS_FILE,
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrModelSpec {
    pub(crate) static_shape: [usize; 4],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrDetectionConfig {
    pub(crate) limit_side_len: u32,
    pub(crate) max_side_limit: u32,
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
    model: OcrDetectionModel,
) -> PpOcrDetectionConfig {
    let db_box_thresh = match model {
        OcrDetectionModel::PpOcrV6Tiny => 0.4,
        OcrDetectionModel::PpOcrV6Small | OcrDetectionModel::PpOcrV6Medium => {
            0.45
        }
    };

    PpOcrDetectionConfig {
        limit_side_len: 736,
        max_side_limit: 4000,
        mean: MEAN,
        std: STD,
        db_thresh: 0.2,
        db_box_thresh,
        db_unclip_ratio: 1.4,
        max_candidates: 3000,
    }
}

pub(crate) fn recognizer_config(
    model: OcrRecognitionModel,
) -> PpOcrRecognitionConfig {
    let num_classes = match model {
        OcrRecognitionModel::PpOcrV6Tiny => 6906,
        OcrRecognitionModel::PpOcrV6Small
        | OcrRecognitionModel::PpOcrV6Medium => 18710,
    };

    PpOcrRecognitionConfig {
        spec: PpOcrModelSpec {
            static_shape: [1, 3, 48, 320],
        },
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        num_classes,
    }
}

/// Returns the bundled upstream dictionary for the given recognizer.
pub(crate) fn recognizer_dictionary(
    model: OcrRecognitionModel,
) -> &'static str {
    match model {
        // PaddlePaddle/PP-OCRv6_tiny_rec_onnx @ 2612ab37152ae0a677521bae4e1e3d4fb4cf7c30
        OcrRecognitionModel::PpOcrV6Tiny => {
            include_str!("assets/tiny_rec_inference.yml")
        }
        // PaddlePaddle/PP-OCRv6_small_rec_onnx @ b8f84f0b80c529de40b4fbb3544b84fa7233a513
        OcrRecognitionModel::PpOcrV6Small => {
            include_str!("assets/small_rec_inference.yml")
        }
        // PaddlePaddle/PP-OCRv6_medium_rec_onnx @ 50c7eacafc52fa7bcf4194e8cd08e46f8558504b
        OcrRecognitionModel::PpOcrV6Medium => {
            include_str!("assets/medium_rec_inference.yml")
        }
    }
}
