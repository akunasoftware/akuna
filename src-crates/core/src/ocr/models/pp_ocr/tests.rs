use image::{Rgb, RgbImage};

use super::{
    preprocess::detector_resize_dims, runtime::crop_box, spec::detector_config,
};
use crate::ocr::OcrDetectionModel;

#[test]
fn detector_configs_match_models() {
    let tiny = detector_config(OcrDetectionModel::PpOcrV6Tiny);
    let small = detector_config(OcrDetectionModel::PpOcrV6Small);
    let medium = detector_config(OcrDetectionModel::PpOcrV6Medium);

    assert_eq!(tiny.limit_side_len, 736);
    assert_eq!(tiny.max_side_limit, 4000);
    assert_eq!(tiny.db_thresh, 0.2);
    assert_eq!(tiny.db_box_thresh, 0.4);
    assert_eq!(tiny.db_unclip_ratio, 1.4);
    assert_eq!(small.db_box_thresh, 0.45);
    assert_eq!(medium.db_box_thresh, 0.45);
}

#[test]
fn detector_resize_applies_minimum_and_maximum() {
    assert_eq!(detector_resize_dims(100, 200, 736, 4000), (736, 1472));
    assert_eq!(detector_resize_dims(1, 10_000, 736, 4000), (32, 4000));
}

#[test]
fn vertical_crops_rotate_counterclockwise() {
    let mut image = RgbImage::new(2, 4);
    image.put_pixel(0, 0, Rgb([255, 0, 0]));

    let crop =
        crop_box(&image, [[0.0, 0.0], [2.0, 0.0], [2.0, 4.0], [0.0, 4.0]])
            .expect("vertical crop should be valid")
            .to_rgb8();

    assert_eq!(crop.dimensions(), (4, 2));
    assert_eq!(crop.get_pixel(0, 1), &Rgb([255, 0, 0]));
}
