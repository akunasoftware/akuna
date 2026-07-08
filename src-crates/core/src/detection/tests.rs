use super::models::magika::sorted_row;
use super::models::magika_preprocess::{PreparedInput, prepare_input};
use super::{DetectionError, DetectionOrigin, FileType};
use crate::detection::vendor::{content::ContentType, model as vendor_model};

#[test]
fn short_utf8_input_is_ruled_as_text() {
    match prepare_input(b"hello", &vendor_model::CONFIG) {
        PreparedInput::Ruled(ContentType::Txt) => {}
        _ => panic!("expected ruled text"),
    }
}

#[test]
fn core_shape_copies_vendor_metadata() {
    let detected = FileType::ruled(ContentType::Txt);
    let expected = ContentType::Txt.info();

    assert_eq!(detected.info().label, expected.label);
    assert_eq!(detected.info().mime_type, expected.mime_type);
    assert_eq!(detected.info().group, expected.group);
    assert_eq!(detected.info().description, expected.description);
    assert_eq!(detected.info().extensions, expected.extensions);
    assert_eq!(detected.info().is_text, expected.is_text);
    assert_eq!(detected.confidence(), 1.0);
    assert_eq!(detected.origin(), DetectionOrigin::Rule);
}

#[test]
fn model_errors_keep_sources() {
    use super::models::magika::MagikaModel;
    use crate::ml::backend::{Backend, cpu_device};

    let error = match MagikaModel::<Backend>::from_bytes(&cpu_device(), &[]) {
        Ok(_) => panic!("invalid weights should fail"),
        Err(error) => error,
    };

    assert!(matches!(error, DetectionError::Model { .. }));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn path_sample_uses_file_edges() {
    let edge_len = vendor_model::CONFIG.block_size;
    let path = std::env::temp_dir()
        .join(format!("akuna-detection-{}.bin", std::process::id()));
    let mut content = vec![b'a'; edge_len];
    content.push(b'm');
    content.extend(vec![b'z'; edge_len]);
    std::fs::write(&path, content).expect("write test file");

    let sample =
        super::detector::read_file_sample(&path).expect("read file sample");
    std::fs::remove_file(path).expect("remove test file");

    assert_eq!(sample.len(), edge_len * 2);
    assert_eq!(&sample[..edge_len], vec![b'a'; edge_len]);
    assert_eq!(&sample[edge_len..], vec![b'z'; edge_len]);
}

#[test]
fn path_sample_keeps_io_source() {
    let path = std::env::temp_dir()
        .join(format!("akuna-missing-detection-{}", std::process::id()));
    let error = super::detector::read_file_sample(path)
        .expect_err("missing input should fail");

    assert!(matches!(error, DetectionError::Io { .. }));
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn tied_scores_use_label_index_order() {
    let mut scores = vec![0.0; vendor_model::NUM_LABELS];
    scores[5] = 0.8;
    scores[2] = 0.8;

    let sorted = sorted_row(scores).expect("valid score row");
    assert_eq!(&sorted[..2], &[(2, 0.8), (5, 0.8)]);
}

#[cfg(all(feature = "extraction", feature = "ocr"))]
#[test]
fn model_result_keeps_confidence_and_origin() {
    use super::models::magika::MagikaModel;
    use crate::detection::DetectionOrigin;
    use crate::ml::backend::{Backend, cpu_device};

    crate::testkit::run_with_model_stack(|| {
        let model = MagikaModel::<Backend>::from_embedded(&cpu_device())?;
        let first =
            model.identify_bytes(b"function greet() { return 'hi'; }")?;
        let second =
            model.identify_bytes(b"function greet() { return 'hi'; }")?;

        assert_eq!(first, second);
        assert_eq!(first.origin(), DetectionOrigin::Model);
        assert!((0.0..=1.0).contains(&first.confidence()));
        Ok(())
    })
    .expect("model test thread should finish");
}
