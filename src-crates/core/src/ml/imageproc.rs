//! Image resizing that matches OpenCV (`cv2.resize`) byte-for-byte, so
//! detection/layout inputs are identical to the PaddleOCR/PaddleX references.

use image::{DynamicImage, RgbImage};

const COEF_BITS: u32 = 11;
const COEF_SCALE: i64 = 1 << COEF_BITS; // 2048

/// Per-output-pixel bilinear taps: `(src0, src1, w0, w1)` at 2048-scale.
fn linear_coeffs(dst: usize, src: usize) -> Vec<(usize, usize, i64, i64)> {
    let scale = src as f64 / dst as f64;
    let coef = COEF_SCALE as f32;
    (0..dst)
        .map(|d| {
            // cv2 casts the mapped coordinate to f32 before the floor/frac.
            let mapped = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let mut s = mapped.floor() as isize;
            let mut frac = mapped - s as f32;
            if s < 0 {
                s = 0;
                frac = 0.0;
            }
            if s >= src as isize - 1 {
                s = src as isize - 1;
                frac = 0.0;
            }
            // cv2 rounds each weight independently (cvRound, half-to-even),
            // entirely in f32 — w0 + w1 is not forced to 2048.
            let w0 = ((1.0f32 - frac) * coef).round_ties_even() as i64;
            let w1 = (frac * coef).round_ties_even() as i64;
            let s0 = s as usize;
            let s1 = (s0 + 1).min(src - 1);
            (s0, s1, w0, w1)
        })
        .collect()
}

/// Saturating cast to the `i16` range.
fn sat16(value: i32) -> i32 {
    value.clamp(-32768, 32767)
}

/// Resizes `image` to `dst_w x dst_h`, matching `cv2.resize(INTER_LINEAR)`
/// byte-for-byte on the RGB8 channels.
pub(crate) fn resize_linear_cv2(
    image: &DynamicImage,
    dst_w: usize,
    dst_h: usize,
) -> RgbImage {
    let src = image.to_rgb8();
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let xc = linear_coeffs(dst_w, sw);
    let yc = linear_coeffs(dst_h, sh);
    let raw = src.as_raw(); // row-major RGB
    let src_stride = sw * 3;
    let dst_stride = dst_w * 3;

    // Horizontal pass: `src * alpha` summed at full int precision (2048-scale).
    let mut hbuf = vec![0i32; sh * dst_stride];
    for y in 0..sh {
        let srow = y * src_stride;
        let hrow = y * dst_stride;
        for (dx, &(sx0, sx1, a0, a1)) in xc.iter().enumerate() {
            let c0 = sx0 * 3;
            let c1 = sx1 * 3;
            let out = hrow + dx * 3;
            for ch in 0..3 {
                let p0 = raw[srow + c0 + ch] as i32;
                let p1 = raw[srow + c1 + ch] as i32;
                hbuf[out + ch] = p0 * a0 as i32 + p1 * a1 as i32;
            }
        }
    }

    // Vertical pass: cv2's SIMD `VResizeLinearVec_32s8u`. Each row is shifted
    // `>> 4` and packed to i16, combined with `mulhi16` (`(a*b) >> 16`) and a
    // saturating add, then `(t + 2) >> 2` and saturated to `u8`.
    let mut out = RgbImage::new(dst_w as u32, dst_h as u32);
    let out_raw = out.as_mut();
    for (dy, &(sy0, sy1, b0, b1)) in yc.iter().enumerate() {
        let r0 = sy0 * dst_stride;
        let r1 = sy1 * dst_stride;
        let orow = dy * dst_stride;
        let (b0, b1) = (b0 as i32, b1 as i32);
        for i in 0..dst_stride {
            let s0 = sat16(hbuf[r0 + i] >> 4);
            let s1 = sat16(hbuf[r1 + i] >> 4);
            let t = sat16(((s0 * b0) >> 16) + ((s1 * b1) >> 16));
            out_raw[orow + i] = ((t + 2) >> 2).clamp(0, 255) as u8;
        }
    }
    out
}

/// Per-output-pixel bicubic taps: 4 source indices + 4 weights (2048-scale).
fn cubic_coeffs(dst: usize, src: usize) -> Vec<([usize; 4], [i64; 4])> {
    // cv2's bicubic kernel parameter.
    const A: f64 = -0.75;
    let scale = src as f64 / dst as f64;
    (0..dst)
        .map(|d| {
            let mapped = (d as f64 + 0.5) * scale - 0.5;
            let s = mapped.floor() as isize;
            let x = mapped - s as f64;
            // cv2 `interpolateCubic`.
            let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A)
                * (x + 1.0)
                - 4.0 * A;
            let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
            let c2 =
                ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x)
                    + 1.0;
            let c3 = 1.0 - c0 - c1 - c2;
            let weights = [c0, c1, c2, c3]
                .map(|c| (c * COEF_SCALE as f64).round_ties_even() as i64);
            let last = src as isize - 1;
            let idx =
                [s - 1, s, s + 1, s + 2].map(|i| i.clamp(0, last) as usize);
            (idx, weights)
        })
        .collect()
}

/// Resizes `image` to `dst_w x dst_h` reproducing `cv2.resize(INTER_CUBIC)`
/// byte-for-byte on the RGB8 channels.
pub(crate) fn resize_cubic_cv2(
    image: &DynamicImage,
    dst_w: usize,
    dst_h: usize,
) -> RgbImage {
    let src = image.to_rgb8();
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let xc = cubic_coeffs(dst_w, sw);
    let yc = cubic_coeffs(dst_h, sh);
    let raw = src.as_raw();
    let row_stride = sw * 3;

    let mut out = RgbImage::new(dst_w as u32, dst_h as u32);
    let out_raw = out.as_mut();
    for (dy, (yi, yw)) in yc.iter().enumerate() {
        for (dx, (xi, xw)) in xc.iter().enumerate() {
            let out_base = (dy * dst_w + dx) * 3;
            for ch in 0..3 {
                // Horizontal 4-tap per source row, then vertical 4-tap.
                let mut v: i64 = 1 << (2 * COEF_BITS - 1);
                for ky in 0..4 {
                    let row = yi[ky] * row_stride;
                    let mut h: i64 = 0;
                    for kx in 0..4 {
                        h += raw[row + xi[kx] * 3 + ch] as i64 * xw[kx];
                    }
                    v += h * yw[ky];
                }
                out_raw[out_base + ch] =
                    (v >> (2 * COEF_BITS)).clamp(0, 255) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    // A deterministic 17x13 RGB source and its `cv2.resize(..., (8, 6))` output
    // for both interpolations, generated by OpenCV (opencv-python). These are
    // byte-for-byte golden vectors guarding the exact fixed-point behaviour
    // (downscaling 17x13 -> 8x6 exercises both passes and the edge taps).
    #[rustfmt::skip]
    const SRC_17X13: &[u8] = &[
        0, 0, 7, 37, 5, 24, 74, 10, 41, 111, 15, 58, 148, 20, 75, 185, 25, 92, 222, 30, 109, 3, 35, 126, 40, 40, 143, 77, 45, 160, 114, 50, 177, 151, 55, 194, 188, 60, 211, 225, 65, 228, 6, 70, 245, 43, 75, 6, 80, 80, 23,
        11, 53, 36, 48, 58, 53, 85, 63, 70, 122, 68, 87, 159, 73, 104, 196, 78, 121, 233, 83, 138, 14, 88, 155, 51, 93, 172, 88, 98, 189, 125, 103, 206, 162, 108, 223, 199, 113, 240, 236, 118, 1, 17, 123, 18, 54, 128, 35, 91, 133, 52,
        22, 106, 65, 59, 111, 82, 96, 116, 99, 133, 121, 116, 170, 126, 133, 207, 131, 150, 244, 136, 167, 25, 141, 184, 62, 146, 201, 99, 151, 218, 136, 156, 235, 173, 161, 252, 210, 166, 13, 247, 171, 30, 28, 176, 47, 65, 181, 64, 102, 186, 81,
        33, 159, 94, 70, 164, 111, 107, 169, 128, 144, 174, 145, 181, 179, 162, 218, 184, 179, 255, 189, 196, 36, 194, 213, 73, 199, 230, 110, 204, 247, 147, 209, 8, 184, 214, 25, 221, 219, 42, 2, 224, 59, 39, 229, 76, 76, 234, 93, 113, 239, 110,
        44, 212, 123, 81, 217, 140, 118, 222, 157, 155, 227, 174, 192, 232, 191, 229, 237, 208, 10, 242, 225, 47, 247, 242, 84, 252, 3, 121, 1, 20, 158, 6, 37, 195, 11, 54, 232, 16, 71, 13, 21, 88, 50, 26, 105, 87, 31, 122, 124, 36, 139,
        55, 9, 152, 92, 14, 169, 129, 19, 186, 166, 24, 203, 203, 29, 220, 240, 34, 237, 21, 39, 254, 58, 44, 15, 95, 49, 32, 132, 54, 49, 169, 59, 66, 206, 64, 83, 243, 69, 100, 24, 74, 117, 61, 79, 134, 98, 84, 151, 135, 89, 168,
        66, 62, 181, 103, 67, 198, 140, 72, 215, 177, 77, 232, 214, 82, 249, 251, 87, 10, 32, 92, 27, 69, 97, 44, 106, 102, 61, 143, 107, 78, 180, 112, 95, 217, 117, 112, 254, 122, 129, 35, 127, 146, 72, 132, 163, 109, 137, 180, 146, 142, 197,
        77, 115, 210, 114, 120, 227, 151, 125, 244, 188, 130, 5, 225, 135, 22, 6, 140, 39, 43, 145, 56, 80, 150, 73, 117, 155, 90, 154, 160, 107, 191, 165, 124, 228, 170, 141, 9, 175, 158, 46, 180, 175, 83, 185, 192, 120, 190, 209, 157, 195, 226,
        88, 168, 239, 125, 173, 0, 162, 178, 17, 199, 183, 34, 236, 188, 51, 17, 193, 68, 54, 198, 85, 91, 203, 102, 128, 208, 119, 165, 213, 136, 202, 218, 153, 239, 223, 170, 20, 228, 187, 57, 233, 204, 94, 238, 221, 131, 243, 238, 168, 248, 255,
        99, 221, 12, 136, 226, 29, 173, 231, 46, 210, 236, 63, 247, 241, 80, 28, 246, 97, 65, 251, 114, 102, 0, 131, 139, 5, 148, 176, 10, 165, 213, 15, 182, 250, 20, 199, 31, 25, 216, 68, 30, 233, 105, 35, 250, 142, 40, 11, 179, 45, 28,
        110, 18, 41, 147, 23, 58, 184, 28, 75, 221, 33, 92, 2, 38, 109, 39, 43, 126, 76, 48, 143, 113, 53, 160, 150, 58, 177, 187, 63, 194, 224, 68, 211, 5, 73, 228, 42, 78, 245, 79, 83, 6, 116, 88, 23, 153, 93, 40, 190, 98, 57,
        121, 71, 70, 158, 76, 87, 195, 81, 104, 232, 86, 121, 13, 91, 138, 50, 96, 155, 87, 101, 172, 124, 106, 189, 161, 111, 206, 198, 116, 223, 235, 121, 240, 16, 126, 1, 53, 131, 18, 90, 136, 35, 127, 141, 52, 164, 146, 69, 201, 151, 86,
        132, 124, 99, 169, 129, 116, 206, 134, 133, 243, 139, 150, 24, 144, 167, 61, 149, 184, 98, 154, 201, 135, 159, 218, 172, 164, 235, 209, 169, 252, 246, 174, 13, 27, 179, 30, 64, 184, 47, 101, 189, 64, 138, 194, 81, 175, 199, 98, 212, 204, 115,
    ];
    #[rustfmt::skip]
    const LINEAR_8X6: &[u8] = &[
        27, 34, 33, 106, 44, 69, 184, 55, 105, 23, 65, 142, 86, 76, 178, 164, 87, 214, 163, 97, 101, 65, 108, 30,
        51, 148, 96, 130, 159, 132, 208, 170, 168, 47, 180, 204, 109, 191, 229, 188, 201, 73, 55, 212, 57, 89, 223, 93,
        75, 29, 159, 153, 39, 195, 232, 50, 231, 55, 61, 47, 133, 50, 48, 212, 60, 84, 35, 71, 120, 113, 82, 156,
        99, 122, 210, 177, 133, 75, 48, 143, 38, 78, 154, 74, 157, 165, 110, 188, 175, 146, 58, 186, 183, 137, 196, 219,
        122, 173, 29, 201, 184, 65, 60, 194, 101, 102, 25, 137, 181, 23, 173, 159, 34, 209, 82, 45, 181, 161, 55, 26,
        146, 96, 91, 225, 106, 128, 48, 117, 164, 126, 128, 200, 205, 138, 229, 27, 149, 16, 106, 160, 52, 185, 170, 88,
    ];
    #[rustfmt::skip]
    const CUBIC_8X6: &[u8] = &[
        23, 28, 29, 103, 39, 66, 182, 50, 102, 18, 60, 139, 86, 71, 176, 165, 82, 220, 168, 93, 82, 68, 103, 18,
        47, 146, 93, 128, 157, 130, 204, 167, 166, 44, 178, 202, 110, 217, 255, 194, 226, 67, 31, 237, 57, 92, 248, 93,
        71, 22, 156, 152, 33, 194, 254, 44, 241, 53, 55, 41, 134, 46, 46, 219, 58, 84, 7, 68, 120, 116, 79, 157,
        95, 123, 212, 176, 134, 63, 28, 145, 36, 78, 157, 75, 158, 168, 112, 193, 178, 148, 62, 189, 184, 141, 200, 223,
        120, 181, 17, 207, 192, 66, 54, 203, 102, 102, 0, 138, 183, 0, 175, 166, 10, 223, 84, 20, 194, 165, 31, 0,
        144, 100, 93, 253, 111, 130, 41, 122, 166, 126, 133, 202, 208, 144, 235, 7, 154, 0, 109, 165, 59, 189, 176, 93,
    ];

    fn src_image() -> DynamicImage {
        let img = RgbImage::from_raw(17, 13, SRC_17X13.to_vec())
            .expect("golden source dimensions match buffer length");
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn linear_matches_cv2_golden() {
        let out = resize_linear_cv2(&src_image(), 8, 6);
        assert_eq!(out.as_raw().as_slice(), LINEAR_8X6, "cv2 INTER_LINEAR");
    }

    #[test]
    fn cubic_matches_cv2_golden() {
        let out = resize_cubic_cv2(&src_image(), 8, 6);
        assert_eq!(out.as_raw().as_slice(), CUBIC_8X6, "cv2 INTER_CUBIC");
    }
}
