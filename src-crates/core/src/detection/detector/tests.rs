use std::sync::mpsc;
use std::sync::{Arc, Barrier};
use std::thread;

use crate::detection::{DetectionOrigin, FileTypeDetector};
use crate::ml::backend::cpu_device;
use crate::testkit::run_with_model_stack;

#[test]
fn default_model_is_shared() {
    let barrier = Arc::new(Barrier::new(4));
    let (sender, receiver) = mpsc::channel();
    thread::scope(|scope| {
        for _ in 0..4 {
            let barrier = Arc::clone(&barrier);
            let sender = sender.clone();
            scope.spawn(move || {
                run_with_model_stack(move || {
                    barrier.wait();
                    let detector = FileTypeDetector::new()?;
                    let result = detector
                        .identify_bytes(b"function greet() { return 'hi'; }")?;
                    sender.send((detector.model, result))?;
                    Ok(())
                })
                .expect("model test thread should finish");
            });
        }
    });
    drop(sender);

    let results: Vec<_> = receiver.into_iter().collect();
    assert_eq!(results.len(), 4);
    let detector = FileTypeDetector::new().expect("default model is loaded");
    for (model, result) in &results {
        assert!(Arc::ptr_eq(model, &detector.model));
        assert_eq!(result, &results[0].1);
        assert_eq!(result.origin(), DetectionOrigin::Model);
        assert!((0.0..=1.0).contains(&result.confidence()));
    }
}

#[test]
fn explicit_device_models_are_not_shared() {
    run_with_model_stack(|| {
        let cached = FileTypeDetector::new()?;
        let first = FileTypeDetector::new_on(cpu_device())?;
        let second = FileTypeDetector::new_on(cpu_device())?;
        assert!(!Arc::ptr_eq(&cached.model, &first.model));
        assert!(!Arc::ptr_eq(&first.model, &second.model));
        Ok(())
    })
    .expect("model test thread should finish");
}
