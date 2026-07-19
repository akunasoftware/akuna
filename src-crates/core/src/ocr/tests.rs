use super::{OcrDetectionModel, OcrEngineOptions, OcrRecognitionModel};
use crate::ocr::models::pp_ocr::postprocess::{TextBox, sort_boxes};
use crate::ocr::models::pp_ocr::{
    dictionary::load_dictionary,
    spec::{detector_weight, recognizer_config, recognizer_weight},
};

#[test]
fn legacy_options_deserialize() {
    let options = serde_json::from_str::<OcrEngineOptions>(
        r#"{"detector":"PpOcrV6TinyDet","recognizer":"PpOcrV6SmallRec"}"#,
    )
    .expect("legacy OCR options should deserialize");

    assert_eq!(options.detection_model, OcrDetectionModel::PpOcrV6Tiny);
    assert_eq!(options.recognition_model, OcrRecognitionModel::PpOcrV6Small);
}

#[test]
fn box_ties_are_stable() {
    let mut boxes = vec![
        TextBox {
            points: [[1.0, 1.0], [4.0, 1.0], [4.0, 3.0], [1.0, 3.0]],
        },
        TextBox {
            points: [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]],
        },
    ];

    sort_boxes(&mut boxes);

    assert_eq!(boxes[0].points[2], [3.0, 3.0]);
    assert_eq!(boxes[1].points[2], [4.0, 3.0]);
}

#[test]
fn bundled_dictionaries_match_recognizer_heads() {
    for model in [
        OcrRecognitionModel::PpOcrV6Tiny,
        OcrRecognitionModel::PpOcrV6Small,
        OcrRecognitionModel::PpOcrV6Medium,
    ] {
        let dictionary =
            load_dictionary(model).expect("bundled dictionary parses");
        let config = recognizer_config(model);

        assert_eq!(dictionary.len() + 1, config.num_classes);
    }
}

#[test]
fn weights_are_pinned() {
    for model in [
        OcrDetectionModel::PpOcrV6Tiny,
        OcrDetectionModel::PpOcrV6Small,
        OcrDetectionModel::PpOcrV6Medium,
    ] {
        let weight = detector_weight(model);
        assert_eq!(weight.revision.len(), 40);
        assert_eq!(weight.filename, "model.safetensors");
    }

    for model in [
        OcrRecognitionModel::PpOcrV6Tiny,
        OcrRecognitionModel::PpOcrV6Small,
        OcrRecognitionModel::PpOcrV6Medium,
    ] {
        let weight = recognizer_weight(model);
        assert_eq!(weight.revision.len(), 40);
        assert_eq!(weight.filename, "model.safetensors");
    }
}
