use anyhow::Result;
use image::{DynamicImage, GenericImageView};

use crate::ocr::models::pp_ocr::spec::{
    PpOcrDetectionConfig, PpOcrRecognitionConfig,
};

/// Upper bound on recognizer input width.
const MAX_RECOGNIZER_WIDTH: usize = 3200;

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
    config: &PpOcrDetectionConfig,
) -> Result<PpOcrInput> {
    let (original_width, original_height) = image.dimensions();
    // PaddleOCR's `DetResizeForTest` with `limit_type="min"`: scale the
    // shortest side up, cap the longest side, then snap each side to 32.
    let (resized_width, resized_height) = detector_resize_dims(
        original_width,
        original_height,
        config.limit_side_len,
        config.max_side_limit,
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

/// Returns the detector input `(width, height)` for an image of the given size.
pub(super) fn detector_resize_dims(
    width: u32,
    height: u32,
    limit_side_len: u32,
    max_side_limit: u32,
) -> (u32, u32) {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let limit = limit_side_len as f64;
    let shortest = w.min(h);
    let ratio = if shortest < limit {
        limit / shortest
    } else {
        1.0
    };
    let mut resized_width = (w * ratio).trunc();
    let mut resized_height = (h * ratio).trunc();
    let longest = resized_width.max(resized_height);
    if longest > max_side_limit as f64 {
        let ratio = max_side_limit as f64 / longest;
        resized_width = (resized_width * ratio).trunc();
        resized_height = (resized_height * ratio).trunc();
    }
    // Python: int(side * ratio) truncates, then round(x / 32) * 32 (ties-even).
    let snap = |scaled: f64| -> u32 {
        let snapped = (scaled / 32.0).round_ties_even() * 32.0;
        (snapped as u32).max(32)
    };
    (snap(resized_width), snap(resized_height))
}

pub(crate) fn preprocess_recognizer(
    image: &DynamicImage,
    config: &PpOcrRecognitionConfig,
) -> Result<PpOcrInput> {
    let (_, _, target_height, nominal_width) =
        static_shape(config.spec.static_shape);
    let (original_width, original_height) = image.dimensions();
    let scale = target_height as f32 / original_height.max(1) as f32;
    let resized_width = ((original_width as f32 * scale).ceil() as usize)
        .clamp(1, MAX_RECOGNIZER_WIDTH);
    let target_width = resized_width.max(nominal_width);
    // cv2 INTER_LINEAR, matching PaddleOCR's `resize_norm_img`.
    let resized = crate::ml::imageproc::resize_linear_cv2(
        image,
        resized_width,
        target_height,
    );
    let values = normalized_nchw(
        &resized,
        target_width,
        target_height,
        config.mean,
        config.std,
    );

    Ok(PpOcrInput {
        values,
        channels: 3,
        height: target_height,
        width: target_width,
        original_width,
        original_height,
    })
}

/// Normalises a Rust RGB image as PaddleOCR BGR into a CHW `f32` tensor.
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
            let values_for_pixel = [pixel[2], pixel[1], pixel[0]];
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
