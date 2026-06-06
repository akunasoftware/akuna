use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use image::{DynamicImage, GenericImageView};
use safetensors::SafeTensors;

use crate::ocr::models::pp_ocr::dictionary::load_dictionary;
use crate::ocr::models::pp_ocr::native::det::PpOcrDetector;
use crate::ocr::models::pp_ocr::native::det_medium::PpOcrDetectorMedium;
use crate::ocr::models::pp_ocr::native::rec::PpOcrRecognizer;
use crate::ocr::models::pp_ocr::native::rec_tiny::PpOcrRecognizerTiny;
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
    OcrBlock, OcrBlockKind, OcrDetector, OcrPage, OcrRecognizer, OcrRect,
};

const CROP_PADDING_RATIO: f32 = 0.08;
const MIN_CROP_PADDING: f32 = 2.0;

#[derive(Debug)]
pub(crate) struct PpOcrRuntime<B: Backend> {
    pub(crate) detector: OcrDetector,
    pub(crate) recognizer: OcrRecognizer,
    detector_model: DetectorModel<B>,
    recognizer_model: RecognizerModel<B>,
    dictionary: Vec<String>,
}

#[derive(Debug)]
enum DetectorModel<B: Backend> {
    Native(Box<PpOcrDetector<B>>),
    NativeMedium(Box<PpOcrDetectorMedium<B>>),
}

#[derive(Debug)]
enum RecognizerModel<B: Backend> {
    Native(Box<PpOcrRecognizer<B>>),
    NativeTiny(Box<PpOcrRecognizerTiny<B>>),
}

impl<B> PpOcrRuntime<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) async fn load(
        detector: OcrDetector,
        recognizer: OcrRecognizer,
        device: &B::Device,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let recognizer_cfg = recognizer_config(recognizer);
        let dictionary =
            load_dictionary(&recognizer_cfg, cache_dir.as_deref()).await?;
        let detector_model = DetectorModel::load(detector, device)?;
        let recognizer_model = RecognizerModel::load(
            recognizer,
            recognizer_cfg.num_classes,
            device,
        )?;

        Ok(Self {
            detector,
            recognizer,
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
        let detector_config = detector_config(self.detector);
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

        let recognizer_config = recognizer_config(self.recognizer);
        let mut blocks = Vec::new();
        for text_box in boxes {
            let crop = crop_box(image, text_box.points)?;
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
            let trimmed = text.text.trim();
            if !trimmed.is_empty() {
                blocks.push(OcrBlock {
                    text: trimmed.to_string(),
                    bbox: OcrRect::from_points(text_box.points),
                    confidence: Some(text.confidence.min(text_box.score)),
                    kind: OcrBlockKind::Text,
                });
            }
        }

        if blocks.is_empty() {
            let text = self.recognize_crop(image, device)?;
            blocks.push(OcrBlock {
                text: text.text,
                bbox: OcrRect {
                    x: 0.0,
                    y: 0.0,
                    width: image.width() as f32,
                    height: image.height() as f32,
                },
                confidence: Some(text.confidence),
                kind: OcrBlockKind::Unknown,
            });
        }

        Ok(OcrPage {
            width: image.width(),
            height: image.height(),
            blocks,
        })
    }

    fn recognize_crop(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<crate::ocr::models::pp_ocr::postprocess::RecognizedText> {
        let recognizer_config = recognizer_config(self.recognizer);
        let recognizer_input =
            preprocess_recognizer(image, &recognizer_config)?;
        let recognizer_tensor = input_tensor(recognizer_input, device);
        let recognizer_output =
            self.recognizer_model.forward(recognizer_tensor);
        let text = postprocess_recognizer(
            recognizer_output,
            &self.dictionary,
            &recognizer_config,
        )?;

        Ok(text)
    }
}

impl<B: Backend<FloatElem = f32>> DetectorModel<B> {
    fn load(detector: OcrDetector, device: &B::Device) -> Result<Self> {
        let bytes = native::fetch_safetensors(det_safetensors_repo(detector))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        match detector {
            // Medium uses the LKPAN detector variant.
            OcrDetector::PpOcrV6MediumDet => Ok(Self::NativeMedium(Box::new(
                PpOcrDetectorMedium::from_safetensors(&tensors, device)?,
            ))),
            OcrDetector::PpOcrV6TinyDet | OcrDetector::PpOcrV6SmallDet => {
                Ok(Self::Native(Box::new(PpOcrDetector::from_safetensors(
                    &tensors,
                    &native::det::det_config(detector),
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
        recognizer: OcrRecognizer,
        num_classes: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let bytes =
            native::fetch_safetensors(rec_safetensors_repo(recognizer))?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        match recognizer {
            // Tiny uses the conv-only head variant.
            OcrRecognizer::PpOcrV6TinyRec => Ok(Self::NativeTiny(Box::new(
                PpOcrRecognizerTiny::from_safetensors(
                    &tensors,
                    num_classes,
                    device,
                )?,
            ))),
            OcrRecognizer::PpOcrV6SmallRec
            | OcrRecognizer::PpOcrV6MediumRec => {
                Ok(Self::Native(Box::new(PpOcrRecognizer::from_safetensors(
                    &tensors,
                    &native::rec::rec_config(recognizer),
                    num_classes,
                    device,
                )?)))
            }
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

fn crop_box(
    image: &DynamicImage,
    points: [[f32; 2]; 4],
) -> Result<DynamicImage> {
    let (image_width, image_height) = image.dimensions();
    let raw_min_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor();
    let raw_min_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor();
    let raw_max_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil();
    let raw_max_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil();

    let crop_width = raw_max_x - raw_min_x;
    let crop_height = raw_max_y - raw_min_y;
    let x_padding = (crop_width * CROP_PADDING_RATIO).max(MIN_CROP_PADDING);
    let y_padding = (crop_height * CROP_PADDING_RATIO).max(MIN_CROP_PADDING);

    let min_x = (raw_min_x - x_padding).clamp(0.0, image_width as f32) as u32;
    let min_y = (raw_min_y - y_padding).clamp(0.0, image_height as f32) as u32;
    let max_x = (raw_max_x + x_padding).clamp(0.0, image_width as f32) as u32;
    let max_y = (raw_max_y + y_padding).clamp(0.0, image_height as f32) as u32;

    if max_x <= min_x || max_y <= min_y {
        anyhow::bail!("PP-OCR detected invalid crop bounds")
    }

    Ok(image.crop_imm(min_x, min_y, max_x - min_x, max_y - min_y))
}
