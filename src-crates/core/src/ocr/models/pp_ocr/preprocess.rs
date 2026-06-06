use anyhow::Result;
use image::{DynamicImage, GenericImageView};

use crate::ocr::models::pp_ocr::spec::{
    PpOcrDetectorConfig, PpOcrRecognizerConfig,
};

/// Safety ceiling for recognizer width (the model is fully width-dynamic but
/// memory grows linearly with width). The trained nominal width is 320, so
/// anything past a few thousand is well outside the trained distribution.
const MAX_RECOGNIZER_WIDTH: usize = 2048;

#[derive(Debug)]
pub(crate) struct PpOcrInput {
    pub(crate) values: Vec<f32>,
    pub(crate) channels: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) original_width: u32,
    pub(crate) original_height: u32,
}

pub(crate) fn preprocess_detector(
    image: &DynamicImage,
    config: &PpOcrDetectorConfig,
) -> Result<PpOcrInput> {
    let (original_width, original_height) = image.dimensions();
    // PaddleOCR's `DetResizeForTest` (limit_type="max"): scale so the longest
    // side fits `limit_side_len`, snap each side to a multiple of 32, resize
    // with cv2 INTER_LINEAR, and feed the variable-size tensor as-is (no pad).
    let (resized_width, resized_height) = detector_resize_dims(
        original_width,
        original_height,
        config.limit_side_len,
    );
    let resized = crate::ml::imageproc::resize_linear_cv2(
        image,
        resized_width as usize,
        resized_height as usize,
    );
    let values = normalized_nchw(
        &resized,
        resized_width as usize,
        resized_height as usize,
        config.mean,
        config.std,
    );

    Ok(PpOcrInput {
        values,
        channels: 3,
        height: resized_height as usize,
        width: resized_width as usize,
        original_width,
        original_height,
    })
}

/// PaddleOCR detector resize target: fit the longest side to `limit_side_len`
/// (only downscaling), then round each side to a multiple of 32 using Python's
/// banker's rounding, with a floor of 32. Returns `(width, height)`.
fn detector_resize_dims(
    width: u32,
    height: u32,
    limit_side_len: u32,
) -> (u32, u32) {
    let (w, h) = (width.max(1) as f32, height.max(1) as f32);
    let longest = w.max(h);
    let ratio = if longest > limit_side_len as f32 {
        limit_side_len as f32 / longest
    } else {
        1.0
    };
    // Python: int(side * ratio) truncates, then round(x / 32) * 32 (ties-even).
    let snap = |side: f32| -> u32 {
        let scaled = (side * ratio).trunc();
        let snapped = (scaled / 32.0).round_ties_even() * 32.0;
        (snapped as u32).max(32)
    };
    (snap(w), snap(h))
}

pub(crate) fn preprocess_recognizer(
    image: &DynamicImage,
    config: &PpOcrRecognizerConfig,
) -> Result<PpOcrInput> {
    let (_, _, target_height, _nominal_width) =
        static_shape(config.spec.static_shape);
    let (original_width, original_height) = image.dimensions();
    let scale = target_height as f32 / original_height.max(1) as f32;
    let resized_width = ((original_width as f32 * scale).ceil() as usize)
        .clamp(1, MAX_RECOGNIZER_WIDTH);
    // cv2 INTER_LINEAR, matching PaddleOCR's `resize_norm_img`.
    let resized = crate::ml::imageproc::resize_linear_cv2(
        image,
        resized_width,
        target_height,
    );
    let values = normalized_nchw(
        &resized,
        resized_width,
        target_height,
        config.mean,
        config.std,
    );

    Ok(PpOcrInput {
        values,
        channels: 3,
        height: target_height,
        width: resized_width,
        original_width,
        original_height,
    })
}

/// Normalises an RGB image into a CHW `f32` tensor (PP-OCRv6 trains on RGB).
fn normalized_nchw(
    rgb: &image::RgbImage,
    target_width: usize,
    target_height: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let mut values = vec![0.0; 3 * target_height * target_width];
    let copy_width = rgb.width().min(target_width as u32) as usize;
    let copy_height = rgb.height().min(target_height as u32) as usize;

    for y in 0..copy_height {
        for x in 0..copy_width {
            let pixel = rgb.get_pixel(x as u32, y as u32).0;
            let values_for_pixel = [pixel[0], pixel[1], pixel[2]];
            for channel in 0..3 {
                let index = channel * target_height * target_width
                    + y * target_width
                    + x;
                values[index] = (values_for_pixel[channel] as f32 / 255.0
                    - mean[channel])
                    / std[channel];
            }
        }
    }

    values
}

fn static_shape(shape: [usize; 4]) -> (usize, usize, usize, usize) {
    (shape[0], shape[1], shape[2], shape[3])
}
