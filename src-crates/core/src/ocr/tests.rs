use std::sync::{Arc, Barrier, mpsc};
use std::thread;

use burn::tensor::Tensor;

use crate::ml::backend::{Backend, cpu_device};
use crate::ocr::models::pp_ocr::postprocess::{
    TextBox, postprocess_recognizer, sort_boxes,
};
use crate::ocr::models::pp_ocr::{
    dictionary::load_dictionary,
    spec::{PpOcrRecognitionConfig, recognizer_config},
};
use crate::ocr::{
    OcrDetectionModel, OcrEngine, OcrEngineOptions, OcrRecognitionModel,
};
use crate::testkit::{corpus_fixture, run_with_model_stack};

#[test]
fn cache_keys_honor_options() -> anyhow::Result<()> {
    let options = OcrEngineOptions::default().effective()?;
    assert_eq!(
        options,
        OcrEngineOptions {
            cache_dir: Some(hf_hub::Cache::default().path().clone()),
            ..OcrEngineOptions::default()
        }
        .effective()?
    );
    let relative = OcrEngineOptions {
        cache_dir: Some("ocr-cache".into()),
        ..OcrEngineOptions::default()
    }
    .effective()?;
    assert_eq!(
        relative.cache_dir,
        Some(std::env::current_dir()?.join("ocr-cache"))
    );
    assert_ne!(
        relative,
        OcrEngineOptions {
            cache_dir: Some("other-ocr-cache".into()),
            ..OcrEngineOptions::default()
        }
        .effective()?
    );
    Ok(())
}

#[test]
fn extracts_heic_bytes() -> anyhow::Result<()> {
    run_with_model_stack(|| {
        let file_path = corpus_fixture("text-hidpi.heic")?;
        let bytes = std::fs::read(file_path)?;
        assert!(crate::heif::has_heif_brand(&bytes));
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async {
                let engine =
                    OcrEngine::new(OcrEngineOptions::default()).await?;
                let page = engine.extract_bytes(&bytes)?;
                assert!(!page.blocks.is_empty());
                Ok(())
            })
    })
}

#[test]
fn default_models_are_shared() -> anyhow::Result<()> {
    run_with_model_stack(|| {
        let file_path = corpus_fixture("text-hidpi.png")?;
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(async {
                let options = OcrEngineOptions {
                    detection_model: OcrDetectionModel::PpOcrV6Tiny,
                    recognition_model: OcrRecognitionModel::PpOcrV6Tiny,
                    ..OcrEngineOptions::default()
                };
                let (first, second) = tokio::join!(
                    OcrEngine::new(options.clone()),
                    OcrEngine::new(options.clone()),
                );
                let first = first?;
                let second = second?;
                assert!(Arc::ptr_eq(&first.model, &second.model));
                let retained = Arc::downgrade(&first.model);
                let barrier = Arc::new(Barrier::new(2));
                let (sender, receiver) = mpsc::channel();
                thread::scope(|scope| {
                    for engine in [first, second] {
                        let barrier = Arc::clone(&barrier);
                        let sender = sender.clone();
                        let file_path = file_path.clone();
                        scope.spawn(move || {
                            run_with_model_stack(move || {
                                barrier.wait();
                                sender.send(engine.extract_file(file_path)?)?;
                                Ok(())
                            })
                            .expect("OCR inference thread should finish");
                        });
                    }
                });
                drop(sender);
                let page = receiver.recv()?;
                assert!(!page.blocks.is_empty());
                assert_eq!(page, receiver.recv()?);
                let retained = retained
                    .upgrade()
                    .expect("cache retains models after callers drop");
                let reloaded = OcrEngine::new(options.clone()).await?;
                assert!(Arc::ptr_eq(&retained, &reloaded.model));
                let explicit = OcrEngine::new_on(cpu_device(), options).await?;
                assert!(!Arc::ptr_eq(&retained, &explicit.model));
                assert_eq!(explicit.device, cpu_device());
                Ok(())
            })
    })
}

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
fn ctc_top1_decodes_both_layouts() -> anyhow::Result<()> {
    let logits = Tensor::<Backend, 3>::from_floats(
        [[
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.10, 0.80, 0.10],
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
            [0.10, 0.80, 0.80],
        ]],
        &cpu_device(),
    );
    let config = PpOcrRecognitionConfig {
        num_classes: 3,
        ..recognizer_config(OcrRecognitionModel::PpOcrV6Tiny)
    };
    let dictionary = ["a".into(), "b".into()];

    // Batched decode must match the single-crop path element for element.
    for base in [logits.clone(), logits.swap_dims(1, 2)] {
        let batch = Tensor::cat(vec![base.clone(), base.clone()], 0);
        for logits in [base, batch] {
            let results = postprocess_recognizer(logits, &dictionary, &config)?;
            for result in &results {
                assert_eq!(result.text, "aaba");
                assert!((result.confidence - 0.825).abs() < 1e-6);
            }
        }
    }
    Ok(())
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
