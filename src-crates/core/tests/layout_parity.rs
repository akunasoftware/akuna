//! Parity tests comparing akuna-core layout detection output against
//! PaddlePaddle's PP-DocLayoutV3 Python reference.
//!
//! These tests spawn `uv run scripts/reference_layout.py` and are therefore
//! marked `#[ignore]` by default. Run with:
//!
//! ```sh
//! cargo test -p akuna-core --features full --test layout_parity -- \
//!     --ignored --nocapture
//! ```

#![cfg(feature = "layout")]

#[path = "common.rs"]
mod common;

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use akuna_core::layout::{LayoutDetector, LayoutOptions};

use crate::common::CorpusFixture;

/// Reference block emitted by `scripts/reference_layout.py`.
#[derive(Debug, Deserialize)]
struct ReferenceBlock {
    label: String,
    #[allow(dead_code)]
    confidence: f32,
    /// x, y, w, h
    bbox: [f32; 4],
    #[allow(dead_code)]
    order: i64,
}

/// Axis-aligned IoU between two (x, y, w, h) boxes.
fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let (ax1, ay1, ax2, ay2) = (a[0], a[1], a[0] + a[2], a[1] + a[3]);
    let (bx1, by1, bx2, by2) = (b[0], b[1], b[0] + b[2], b[1] + b[3]);
    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);
    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter = inter_w * inter_h;
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// IoU threshold for matching an actual block to a reference block of the same
/// label. With byte-exact cv2 INTER_CUBIC preprocessing the native detector
/// reproduces PaddleX coordinates to IoU mean 0.984 / min 0.946 on the fixture;
/// 0.85 sits comfortably below that observed floor while leaving headroom for
/// cross-device wgpu float ordering, and is tight enough that any genuinely
/// divergent box (which would land far lower) fails.
const IOU_THRESHOLD: f32 = 0.85;

/// Fraction of unmatched blocks tolerated per side. The detector now matches
/// the reference block-for-block, so we require an exact correspondence.
const UNMATCHED_TOLERANCE: f64 = 0.0;

/// Runs PaddleX PP-DocLayoutV3 against `image` and returns its blocks.
fn reference_blocks(image: &std::path::Path) -> Result<Vec<ReferenceBlock>> {
    let image_str = image.to_str().context("non-utf8 fixture path")?;
    common::run_uv_script_args("reference_layout.py", &["--image", image_str])
}

/// Loads our layout detector, runs it on `fixture`, and asserts parity
/// against the PaddleX reference.
///
/// Parity rules:
/// - For each actual block, find the highest-IoU same-label reference block
/// - Count as matched if IoU ≥ [`IOU_THRESHOLD`]
/// - Allow up to [`UNMATCHED_TOLERANCE`] × max(actual, expected) unmatched
///   on each side (covers NMS / threshold drift between runtimes)
fn layout_parity(fixture: CorpusFixture) -> Result<()> {
    let runtime =
        tokio::runtime::Runtime::new().context("tokio runtime should start")?;

    runtime.block_on(async {
        let image_path = fixture.get()?;
        let image = image::open(&image_path).with_context(|| {
            format!("failed to open fixture {}", image_path.display())
        })?;

        let detector = LayoutDetector::new(LayoutOptions::default())
            .await
            .context("layout model should load")?;
        let page = detector
            .detect_image(&image)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("layout detection should succeed")?;

        let actual: Vec<(String, [f32; 4])> = page
            .blocks
            .iter()
            .map(|b| {
                (
                    b.label.clone(),
                    [b.bbox.x, b.bbox.y, b.bbox.width, b.bbox.height],
                )
            })
            .collect();
        let reference =
            reference_blocks(&image_path).context("reference layout failed")?;
        let expected: Vec<(String, [f32; 4])> = reference
            .iter()
            .map(|b| (b.label.clone(), b.bbox))
            .collect();

        let actual_labels: HashSet<&str> =
            actual.iter().map(|(l, _)| l.as_str()).collect();
        let expected_labels: HashSet<&str> =
            expected.iter().map(|(l, _)| l.as_str()).collect();
        let label_set_diff: Vec<&&str> = actual_labels
            .symmetric_difference(&expected_labels)
            .collect();

        if !label_set_diff.is_empty() {
            bail!(
                "layout label set mismatch for {fixture}\n\
                 actual labels:   {actual_labels:?}\n\
                 reference labels: {expected_labels:?}\n\
                 symmetric diff: {label_set_diff:?}"
            );
        }

        let mut unmatched_actual = 0usize;
        let mut consumed = vec![false; expected.len()];
        for (a_label, a_bbox) in &actual {
            let mut best_iou = 0.0f32;
            let mut best_idx = None;
            for (idx, (e_label, e_bbox)) in expected.iter().enumerate() {
                if consumed[idx] || e_label != a_label {
                    continue;
                }
                let score = iou(*a_bbox, *e_bbox);
                if score > best_iou {
                    best_iou = score;
                    best_idx = Some(idx);
                }
            }
            match best_idx {
                Some(idx) if best_iou >= IOU_THRESHOLD => consumed[idx] = true,
                _ => unmatched_actual += 1,
            }
        }
        let unmatched_expected = consumed.iter().filter(|c| !**c).count();

        let max_unmatched = (actual.len().max(expected.len()) as f64
            * UNMATCHED_TOLERANCE)
            .ceil() as usize;
        let ok = unmatched_actual <= max_unmatched
            && unmatched_expected <= max_unmatched;

        if !ok {
            bail!(
                "layout parity failed for {fixture}\n\
                 actual blocks:   {actual_len} (unmatched: {unmatched_actual})\n\
                 reference blocks: {expected_len} (unmatched: {unmatched_expected})\n\
                 allowed unmatched per side: {max_unmatched}\n\
                 iou threshold: {IOU_THRESHOLD}",
                actual_len = actual.len(),
                expected_len = expected.len(),
            );
        }

        Ok(())
    })
}

#[test]
#[ignore = "downloads model and runs Python PaddleX reference"]
fn parity_pp_doclayout_v3_matches_paddlex() {
    let result = run_with_model_stack(|| {
        layout_parity(CorpusFixture::new("content/fixtures/text-hidpi.png"))
    });
    if let Err(error) = result {
        eprintln!("PP-DocLayoutV3 parity failed: {error:?}");
        std::process::exit(1);
    }
}

/// Layout inference needs a large stack; wrap each parity body in a thread
/// with its own tokio runtime. The worker returns its `Result` so panics
/// happen in the caller (outside the wgpu FFI unwind path).
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
