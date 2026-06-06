/// OCR output for one image/page.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct OcrPage {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// OCR blocks in reading order.
    pub blocks: Vec<OcrBlock>,
}

/// OCR text block with geometry and confidence.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct OcrBlock {
    /// Recognized text.
    pub text: String,
    /// Bounding box in image pixel coordinates.
    pub bbox: OcrRect,
    /// Recognition confidence from 0 to 1 when available.
    pub confidence: Option<f32>,
    /// OCR-local block kind.
    pub kind: OcrBlockKind,
}

/// Axis-aligned OCR bounding box.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct OcrRect {
    /// Left coordinate in pixels.
    pub x: f32,
    /// Top coordinate in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
}

impl OcrRect {
    /// Builds a bounding rectangle from four points.
    pub(crate) fn from_points(points: [[f32; 2]; 4]) -> Self {
        let min_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min);
        let min_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min);
        let max_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let max_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max);

        Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

/// OCR-local block classification.
#[non_exhaustive]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub enum OcrBlockKind {
    /// Regular text.
    Text,
    /// Unknown OCR block kind.
    Unknown,
}
