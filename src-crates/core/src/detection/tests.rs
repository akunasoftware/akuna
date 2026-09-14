use super::models::magika::best_score;
use super::models::magika_preprocess;
use super::{DetectionError, DetectionOrigin, FileType};
use crate::detection::vendor::{content::ContentType, model as vendor_model};

#[test]
fn short_utf8_input_is_ruled_as_text() {
    match magika_preprocess::prepare_input(b"hello", &vendor_model::CONFIG) {
        magika_preprocess::PreparedInput::Ruled(ContentType::Txt) => {}
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
fn known_utf8_extensions_are_ruled() {
    for extension in ["md", ".MD"] {
        let detected = super::FileTypeDetector::identify_utf8_extension(
            extension,
            b"valid utf-8",
        )
        .unwrap_or_else(|| panic!("{extension} should resolve"));

        assert_eq!(detected.info().label, "markdown", "{extension}");
        assert_eq!(detected.origin(), DetectionOrigin::Rule, "{extension}");
        assert_eq!(detected.confidence(), 1.0, "{extension}");
    }

    let nix = super::FileTypeDetector::identify_utf8_extension(
        "nix",
        b"{ pkgs, ... }: { }",
    )
    .expect("Nix extension should resolve");

    assert_eq!(nix.info().label, "nix");
    assert_eq!(nix.info().mime_type, "text/x-nix");
    assert_eq!(nix.info().group, "code");
    assert_eq!(nix.info().description, "Nix source");
    assert_eq!(nix.info().extensions, vec!["nix".to_string()]);
    assert!(nix.info().is_text);
    assert_eq!(nix.origin(), DetectionOrigin::Rule);
    assert_eq!(nix.confidence(), 1.0);
    let identify = super::FileTypeDetector::identify_utf8_extension;
    assert!(identify("xml", b"<root />").is_none());
    assert!(identify("md", b"valid prefix\xff").is_none());
}

#[test]
fn model_errors_keep_sources() {
    use super::models::magika::MagikaModel;
    use crate::ml::backend;

    let error = match MagikaModel::<backend::Backend>::from_bytes(
        &backend::cpu_device(),
        &[],
    ) {
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
fn path_sample_rejects_non_files() {
    let error = super::detector::read_file_sample(std::env::temp_dir())
        .expect_err("directory input should fail");

    assert!(matches!(
        error,
        DetectionError::Io { source }
            if source.kind() == std::io::ErrorKind::InvalidInput
    ));
}

#[test]
fn tied_scores_use_label_index_order() {
    let mut scores = vec![0.0; vendor_model::NUM_LABELS];
    scores[5] = 0.8;
    scores[2] = 0.8;

    assert_eq!(best_score(&scores).expect("valid score row"), (2, 0.8));
}

#[test]
fn invalid_scores_are_rejected() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
        let mut scores = vec![0.0; vendor_model::NUM_LABELS];
        scores[0] = invalid;

        assert!(matches!(
            best_score(&scores),
            Err(DetectionError::InvalidModel { .. })
        ));
    }
}
