//! Parity tests comparing akuna-core OCR output against PaddleOCR run via
//! `scripts/reference_ocr.py`.
//!
//! Each test downloads model checkpoints and spawns a Python reference, so all
//! are marked `#[ignore]`. Run one with:
//!
//! ```sh
//! cargo test -p akuna-core --features full --test ocr_parity -- \
//!     --nocapture --ignored parity_pp_ocr_v6_small_matches_paddleocr
//! ```
#![cfg(feature = "ocr")]

#[path = "common.rs"]
mod common;

use anyhow::{Context, Result, bail};

use akuna_core::ocr::{Ocr, OcrDetector, OcrOptions, OcrRecognizer};

use crate::common::CorpusFixture;

/// Reference block emitted by `scripts/reference_ocr.py`.
#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct ReferenceBlock {
    text: String,
    bbox: [f32; 4],
    confidence: f32,
}

#[test]
#[ignore = "downloads model and runs Python PaddleOCR reference"]
fn parity_pp_ocr_v6_tiny_matches_paddleocr() {
    let result = run_with_model_stack(|| {
        text_parity(
            OcrDetector::PpOcrV6TinyDet,
            OcrRecognizer::PpOcrV6TinyRec,
            "tiny",
            &CorpusFixture::new("content/fixtures/text-hidpi.png"),
        )
    });
    if let Err(error) = result {
        eprintln!("PP-OCRv6 tiny parity failed: {error}");
        std::process::exit(1);
    }
}

#[test]
#[ignore = "downloads model and runs Python PaddleOCR reference"]
fn parity_pp_ocr_v6_small_matches_paddleocr() {
    let result = run_with_model_stack(|| {
        text_parity(
            OcrDetector::PpOcrV6SmallDet,
            OcrRecognizer::PpOcrV6SmallRec,
            "small",
            &CorpusFixture::new("content/fixtures/text-hidpi.png"),
        )
    });
    if let Err(error) = result {
        eprintln!("PP-OCRv6 small parity failed: {error}");
        std::process::exit(1);
    }
}

#[test]
#[ignore = "downloads model and runs Python PaddleOCR reference"]
fn parity_pp_ocr_v6_medium_matches_paddleocr() {
    let result = run_with_model_stack(|| {
        text_parity(
            OcrDetector::PpOcrV6MediumDet,
            OcrRecognizer::PpOcrV6MediumRec,
            "medium",
            &CorpusFixture::new("content/fixtures/text-hidpi.png"),
        )
    });
    if let Err(error) = result {
        eprintln!("PP-OCRv6 medium parity failed: {error}");
        std::process::exit(1);
    }
}

/// Compares our OCR output against the Python reference for a detector +
/// recognizer pair.
///
/// Returns `Ok(())` on parity, `Err(message)` on mismatch. Returning a
/// `Result` avoids panicking inside the spawned worker thread, where wgpu
/// cleanup during unwind can abort the process.
fn text_parity(
    detector: OcrDetector,
    recognizer: OcrRecognizer,
    tier: &str,
    fixture: &CorpusFixture,
) -> Result<()> {
    let runtime =
        tokio::runtime::Runtime::new().context("tokio runtime should start")?;
    runtime.block_on(async {
        let image_path = fixture.get()?;

        let ocr = Ocr::new(OcrOptions {
            detector,
            recognizer,
            ..Default::default()
        })
        .await
        .context("ocr model should load")?;

        let page = ocr
            .extract_page_file(&image_path)
            .context("ocr extraction should succeed")?;

        let actual: Vec<String> =
            page.blocks.iter().map(|b| b.text.clone()).collect();
        let reference: Vec<ReferenceBlock> = common::run_uv_script_args(
            "reference_ocr.py",
            &[
                "--image",
                image_path.to_str().context("non-utf8 fixture path")?,
                "--tier",
                tier,
            ],
        )
        .context("reference ocr failed")?;
        let expected: Vec<String> =
            reference.iter().map(|b| b.text.clone()).collect();

        let actual_text = common::aggregated_text(&actual);
        let expected_text = common::aggregated_text(&expected);
        let similarity = common::similarity(&actual_text, &expected_text);

        // With byte-exact cv2 INTER_LINEAR preprocessing the native pipeline
        // reproduces PaddleOCR's text essentially exactly: small/medium reach
        // 1.0 similarity on the fixture, tiny reaches 0.9957 (the smallest
        // model carries genuine per-character recognition noise). Hold the
        // strong tiers to near-exact and tiny to just under its measured floor;
        // a real regression drops similarity far below either bar.
        let parity_threshold: f64 = match tier {
            "tiny" => 0.995,
            _ => 0.999,
        };
        if similarity < parity_threshold {
            bail!(
                "PP-OCRv6 {tier} text parity failed for {fixture}\n\
                 similarity: {similarity:.4} (threshold {parity_threshold})\n\
                 actual blocks:   {actual:?}\n\
                 reference blocks: {expected:?}"
            );
        }

        Ok(())
    })
}

/// OCR inference needs a large stack; wrap each parity body in a thread with
/// its own tokio runtime. The worker returns its `Result` so panics happen in
/// the caller (outside the wgpu FFI unwind path).
fn run_with_model_stack<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
    let handle = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(move || {
            let _ = tx.send(f());
        })
        .context("failed to spawn parity thread")?;
    handle.join().ok();
    rx.recv().context("parity thread dropped result")?
}
