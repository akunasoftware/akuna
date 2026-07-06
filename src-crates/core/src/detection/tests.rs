use super::models::magika::MagikaModel;
use super::models::magika_preprocess::{PreparedInput, prepare_input};
use crate::detection::vendor::{content::ContentType, model as vendor_model};
use crate::ml::backend::{Backend, cpu_device};

/// One ranked label guess produced by the classifier.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RankedAlternative {
    pub label: String,
    pub mime_type: Option<String>,
    pub confidence: f32,
}

/// Top-level result of classifying a single input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Detection {
    pub label: String,
    pub mime_type: Option<String>,
    pub confidence: f32,
    pub alternatives: Vec<RankedAlternative>,
}

/// Builds a full test detection for one content type.
pub(crate) fn detection_for_content_type(
    content_type: ContentType,
) -> Detection {
    let alternative = alternative_for_content_type(content_type, 1.0);

    Detection {
        label: alternative.label.clone(),
        mime_type: alternative.mime_type.clone(),
        confidence: alternative.confidence,
        alternatives: vec![alternative],
    }
}

/// Builds one ranked test alternative for one content type.
pub(crate) fn alternative_for_content_type(
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

#[test]
fn short_utf8_input_is_ruled_as_text() {
    match prepare_input(b"hello".as_slice(), &vendor_model::CONFIG) {
        PreparedInput::Ruled(ContentType::Txt) => {}
        _ => panic!("expected ruled text"),
    }
}

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
fn explicit_top_k_is_applied() {
    let classifier = MagikaModel::<Backend>::from_embedded(&cpu_device())
        .expect("build model")
        .with_top_k(5);

    let detection = classifier
        .detect_bytes(b"function greet() { return 'hi'; }")
        .expect("detect bytes");
    assert_eq!(detection.alternatives.len(), 5);
}
