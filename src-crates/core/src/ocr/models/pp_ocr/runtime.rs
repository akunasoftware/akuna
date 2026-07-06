use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use image::{DynamicImage, Rgb, RgbImage};
use safetensors::SafeTensors;

use crate::ocr::models::pp_ocr::dictionary::load_dictionary;
use crate::ocr::models::pp_ocr::native::det::PpOcrTextDetector;
use crate::ocr::models::pp_ocr::native::det_medium::PpOcrTextDetectorMedium;
use crate::ocr::models::pp_ocr::native::rec::PpOcrTextRecognizer;
use crate::ocr::models::pp_ocr::native::rec_tiny::PpOcrTextRecognizerTiny;
use crate::ocr::models::pp_ocr::native::{self};
use crate::ocr::models::pp_ocr::postprocess::{
    postprocess_detector, postprocess_recognizer,
};
use crate::ocr::models::pp_ocr::preprocess::{
    PpOcrInput, preprocess_detector, preprocess_recognizer,
};
use crate::ocr::models::pp_ocr::spec::{
    det_safetensors_repo, detector_config, rec_safetensors_repo,
    recognizer_config,
};
use crate::ocr::{
    OcrBlock, OcrBlockKind, OcrDetectionModel, OcrPage, OcrRecognitionModel,
    OcrRect,
};

#[derive(Debug)]
pub(crate) struct PpOcrRuntime<B: Backend> {
    pub(crate) detection_model: OcrDetectionModel,
    pub(crate) recognition_model: OcrRecognitionModel,
    detector_model: DetectorModel<B>,
    recognizer_model: RecognizerModel<B>,
    dictionary: Vec<String>,
}

#[derive(Debug)]
enum DetectorModel<B: Backend> {
    Native(Box<PpOcrTextDetector<B>>),
    NativeMedium(Box<PpOcrTextDetectorMedium<B>>),
}

#[derive(Debug)]
enum RecognizerModel<B: Backend> {
    Native(Box<PpOcrTextRecognizer<B>>),
    NativeTiny(Box<PpOcrTextRecognizerTiny<B>>),
}

impl<B> PpOcrRuntime<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) async fn load(
        detection_model: OcrDetectionModel,
        recognition_model: OcrRecognitionModel,
        device: &B::Device,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let recognizer_cfg = recognizer_config(recognition_model);
        let dictionary =
            load_dictionary(&recognizer_cfg, cache_dir.as_deref()).await?;
        let detector_model = DetectorModel::load(detection_model, device)?;
        let recognizer_model = RecognizerModel::load(
            recognition_model,
            recognizer_cfg.num_classes,
            device,
        )?;

        Ok(Self {
            detection_model,
            recognition_model,
            detector_model,
            recognizer_model,
            dictionary,
        })
    }

    pub(crate) fn extract_page(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<OcrPage> {
        let detector_config = detector_config(self.detection_model);
        let detector_input = preprocess_detector(image, &detector_config)?;
        let original_width = detector_input.original_width;
        let original_height = detector_input.original_height;
        let detector_tensor = input_tensor(detector_input, device);
        let detector_output = self.detector_model.forward(detector_tensor);
        let boxes = postprocess_detector(
            detector_output,
            &detector_config,
            original_width,
            original_height,
        )?;

        let recognizer_config = recognizer_config(self.recognition_model);
        let rgb_image = image.to_rgb8();
        let mut blocks = Vec::new();
        for text_box in boxes {
            let crop = crop_box(&rgb_image, text_box.points)?;
            let recognizer_input =
                preprocess_recognizer(&crop, &recognizer_config)?;
            let recognizer_tensor = input_tensor(recognizer_input, device);
            let recognizer_output =
                self.recognizer_model.forward(recognizer_tensor);
            let text = postprocess_recognizer(
                recognizer_output,
                &self.dictionary,
                &recognizer_config,
            )?;
            // Drop boxes the recognizer reads as empty.
            if !text.text.trim().is_empty() {
                blocks.push(OcrBlock {
                    text: text.text,
                    bbox: OcrRect::from_points(text_box.points),
                    confidence: Some(text.confidence),
                    kind: OcrBlockKind::Text,
                });
            }
        }

        Ok(OcrPage {
            width: image.width(),
            height: image.height(),
            blocks,
        })
    }
}

impl<B: Backend<FloatElem = f32>> DetectorModel<B> {
    fn load(model: OcrDetectionModel, device: &B::Device) -> Result<Self> {
        let bytes = native::fetch_safetensors(det_safetensors_repo(model))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        match model {
            // Medium uses the LKPAN detector variant.
            OcrDetectionModel::PpOcrV6Medium => Ok(Self::NativeMedium(
                Box::new(PpOcrTextDetectorMedium::from_safetensors(
                    &tensors, device,
                )?),
            )),
            OcrDetectionModel::PpOcrV6Tiny
            | OcrDetectionModel::PpOcrV6Small => {
                Ok(Self::Native(Box::new(PpOcrTextDetector::from_safetensors(
                    &tensors,
                    &native::det::det_config(model),
                    device,
                )?)))
            }
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Native(model) => model.forward(input),
            Self::NativeMedium(model) => model.forward(input),
        }
    }
}

impl<B: Backend<FloatElem = f32>> RecognizerModel<B> {
    fn load(
        model: OcrRecognitionModel,
        num_classes: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let bytes = native::fetch_safetensors(rec_safetensors_repo(model))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        match model {
            // Tiny uses the conv-only head variant.
            OcrRecognitionModel::PpOcrV6Tiny => Ok(Self::NativeTiny(Box::new(
                PpOcrTextRecognizerTiny::from_safetensors(
                    &tensors,
                    num_classes,
                    device,
                )?,
            ))),
            OcrRecognitionModel::PpOcrV6Small
            | OcrRecognitionModel::PpOcrV6Medium => Ok(Self::Native(Box::new(
                PpOcrTextRecognizer::from_safetensors(
                    &tensors,
                    &native::rec::rec_config(model),
                    num_classes,
                    device,
                )?,
            ))),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        match self {
            Self::Native(model) => model.forward(input),
            Self::NativeTiny(model) => model.forward(input),
        }
    }
}

fn input_tensor<B: Backend>(
    input: PpOcrInput,
    device: &B::Device,
) -> Tensor<B, 4> {
    let data = TensorData::new(
        input.values,
        [1, input.channels, input.height, input.width],
    );

    Tensor::<B, 4>::from_data(data, device)
}

fn crop_box(image: &RgbImage, points: [[f32; 2]; 4]) -> Result<DynamicImage> {
    // The current detector is closest to PaddleOCR parity with truncated boxes.
    let points =
        points.map(|point| [point[0] as i32 as f32, point[1] as i32 as f32]);
    let crop_width = edge_len(points[0], points[1])
        .max(edge_len(points[2], points[3])) as u32;
    let crop_height = edge_len(points[0], points[3])
        .max(edge_len(points[1], points[2])) as u32;
    if crop_width == 0 || crop_height == 0 {
        anyhow::bail!("PP-OCR detected invalid crop bounds")
    }

    let mut crop = warp_crop(image, points, crop_width, crop_height);
    if crop.height() as f32 / crop.width() as f32 >= 1.5 {
        crop = image::imageops::rotate90(&crop);
    }

    Ok(DynamicImage::ImageRgb8(crop))
}

fn edge_len(start: [f32; 2], end: [f32; 2]) -> f32 {
    ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt()
}

fn warp_crop(
    image: &RgbImage,
    points: [[f32; 2]; 4],
    width: u32,
    height: u32,
) -> RgbImage {
    let mut crop = RgbImage::new(width, height);
    for y in 0..height {
        // OpenCV maps destination pixels against a [0, width] x [0, height]
        // rectangle, so the last sampled pixel does not reach 1.0 exactly.
        let v = y as f32 / height as f32;
        for x in 0..width {
            let u = x as f32 / width as f32;
            let top = lerp_point(points[0], points[1], u);
            let bottom = lerp_point(points[3], points[2], u);
            let source = lerp_point(top, bottom, v);
            crop.put_pixel(x, y, sample_cubic(image, source[0], source[1]));
        }
    }
    crop
}

fn lerp_point(start: [f32; 2], end: [f32; 2], t: f32) -> [f32; 2] {
    [
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
    ]
}

fn sample_cubic(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let base_x = x.floor() as i32;
    let base_y = y.floor() as i32;
    let weights_x = cubic_weights(x - base_x as f32);
    let weights_y = cubic_weights(y - base_y as f32);
    let mut channels = [0.0; 3];

    for (ky, weight_y) in weights_y.into_iter().enumerate() {
        let source_y =
            (base_y + ky as i32 - 1).clamp(0, image.height() as i32 - 1) as u32;
        for (kx, weight_x) in weights_x.into_iter().enumerate() {
            let source_x = (base_x + kx as i32 - 1)
                .clamp(0, image.width() as i32 - 1)
                as u32;
            let weight = weight_x * weight_y;
            let pixel = image.get_pixel(source_x, source_y).0;
            for channel in 0..3 {
                channels[channel] += pixel[channel] as f32 * weight;
            }
        }
    }

    Rgb(channels.map(|value| value.round().clamp(0.0, 255.0) as u8))
}

fn cubic_weights(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 =
        ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    let c3 = 1.0 - c0 - c1 - c2;
    [c0, c1, c2, c3]
}
