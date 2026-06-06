use anyhow::Result;
use burn::tensor::{Tensor, backend::Backend};

use crate::ocr::models::pp_ocr::spec::{
    PpOcrDetectorConfig, PpOcrRecognizerConfig,
};

const MIN_COMPONENT_AREA_RATIO: f32 = 0.00002;
const MIN_COMPONENT_SIDE: usize = 3;

#[derive(Debug, Clone)]
pub(crate) struct TextBox {
    pub(crate) points: [[f32; 2]; 4],
    pub(crate) score: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct RecognizedText {
    pub(crate) text: String,
    pub(crate) confidence: f32,
}

pub(crate) fn postprocess_detector<B: Backend<FloatElem = f32>>(
    prob_map: Tensor<B, 4>,
    config: &PpOcrDetectorConfig,
    original_width: u32,
    original_height: u32,
) -> Result<Vec<TextBox>> {
    let [batch, channels, height, width] = prob_map.dims();
    let values = prob_map.into_data().to_vec::<f32>()?;
    if batch == 0 || channels == 0 || height == 0 || width == 0 {
        return Ok(Vec::new());
    }

    // Box coordinates live in the probability map's space (= the resized input
    // dimensions), so map straight back to the original by the actual map size.
    // This is correct regardless of the resize strategy used upstream.
    let x_scale = original_width as f32 / width as f32;
    let y_scale = original_height as f32 / height as f32;

    let map_len = height * width;
    let mut visited = vec![false; map_len];
    let mut boxes = Vec::new();

    for start in 0..map_len {
        if visited[start] || values[start] < config.db_thresh {
            continue;
        }

        let mut stack = vec![start];
        visited[start] = true;
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut score_sum = 0.0;
        let mut count = 0usize;

        while let Some(index) = stack.pop() {
            let x = index % width;
            let y = index / width;
            let score = values[index];
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            score_sum += score;
            count += 1;

            for (next_x, next_y) in neighbors(x, y, width, height) {
                let next = next_y * width + next_x;
                if visited[next] || values[next] < config.db_thresh {
                    continue;
                }
                visited[next] = true;
                stack.push(next);
            }
        }

        let score = score_sum / count.max(1) as f32;
        if score < config.db_box_thresh {
            continue;
        }

        let component_width = max_x + 1 - min_x;
        let component_height = max_y + 1 - min_y;
        let min_area =
            (map_len as f32 * MIN_COMPONENT_AREA_RATIO).ceil() as usize;
        if count < min_area
            || component_width < MIN_COMPONENT_SIDE
            || component_height < MIN_COMPONENT_SIDE
        {
            continue;
        }

        let (left, top, right, bottom) = expand_rect(
            min_x as f32,
            min_y as f32,
            (max_x + 1) as f32,
            (max_y + 1) as f32,
            config.db_unclip_ratio,
        );
        let right_limit = width as f32;
        let bottom_limit = height as f32;
        let left = left.clamp(0.0, right_limit) * x_scale;
        let top = top.clamp(0.0, bottom_limit) * y_scale;
        let right = right.clamp(0.0, right_limit) * x_scale;
        let bottom = bottom.clamp(0.0, bottom_limit) * y_scale;

        if right <= left || bottom <= top {
            continue;
        }

        boxes.push(TextBox {
            points: [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            score,
        });
        if boxes.len() >= config.max_candidates {
            break;
        }
    }

    boxes.sort_by(reading_order);
    Ok(boxes)
}

pub(crate) fn postprocess_recognizer<B: Backend<FloatElem = f32>>(
    logits: Tensor<B, 3>,
    dictionary: &[String],
    config: &PpOcrRecognizerConfig,
) -> Result<RecognizedText> {
    let [_batch, dim1, dim2] = logits.dims();
    let values = logits.into_data().to_vec::<f32>()?;
    let classes = if dim2 == config.num_classes {
        dim2
    } else {
        dim1
    };
    let steps = if dim2 == config.num_classes {
        dim1
    } else {
        dim2
    };
    if classes == 0 || steps == 0 {
        return Ok(RecognizedText {
            text: String::new(),
            confidence: 0.0,
        });
    }

    let mut indices = Vec::with_capacity(steps);
    let mut confidence_sum = 0.0;
    for step in 0..steps {
        let mut best_index = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for class_index in 0..classes {
            let value_index = if dim2 == config.num_classes {
                step * classes + class_index
            } else {
                class_index * steps + step
            };
            let score = values[value_index];
            if score > best_score {
                best_score = score;
                best_index = class_index;
            }
        }
        indices.push(best_index);
        confidence_sum += best_score;
    }

    Ok(RecognizedText {
        text: ctc_decode_indices(indices, dictionary),
        confidence: confidence_sum / steps as f32,
    })
}

fn neighbors(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = (usize, usize)> {
    [
        x.checked_sub(1).map(|next_x| (next_x, y)),
        (x + 1 < width).then_some((x + 1, y)),
        y.checked_sub(1).map(|next_y| (x, next_y)),
        (y + 1 < height).then_some((x, y + 1)),
    ]
    .into_iter()
    .flatten()
}

fn expand_rect(
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    ratio: f32,
) -> (f32, f32, f32, f32) {
    let center_x = (left + right) * 0.5;
    let center_y = (top + bottom) * 0.5;
    let half_width = (right - left) * ratio * 0.5;
    let half_height = (bottom - top) * ratio * 0.5;

    (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
}

fn reading_order(left: &TextBox, right: &TextBox) -> std::cmp::Ordering {
    let (left_x, left_y, _left_width, left_height) = box_bounds(left);
    let (right_x, right_y, _right_width, right_height) = box_bounds(right);
    let line_tolerance = (left_height.min(right_height) * 0.5).max(8.0);

    if (left_y - right_y).abs() <= line_tolerance {
        return left_x.total_cmp(&right_x);
    }

    left_y.total_cmp(&right_y)
}

fn box_bounds(text_box: &TextBox) -> (f32, f32, f32, f32) {
    let min_x = text_box
        .points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min);
    let min_y = text_box
        .points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min);
    let max_x = text_box
        .points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = text_box
        .points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max);

    (min_x, min_y, max_x - min_x, max_y - min_y)
}

pub(crate) fn ctc_decode_indices(
    indices: impl IntoIterator<Item = usize>,
    dictionary: &[String],
) -> String {
    let mut previous = None;
    let mut text = String::new();
    for index in indices {
        if index == 0 || Some(index) == previous {
            previous = Some(index);
            continue;
        }
        if let Some(value) = dictionary.get(index - 1) {
            text.push_str(value);
        }
        previous = Some(index);
    }
    text
}
