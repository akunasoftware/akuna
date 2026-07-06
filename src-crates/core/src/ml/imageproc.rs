//! Image resizing for model preprocessing.

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

/// Resizes an image with linear interpolation.
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

/// Resizes an image with cubic interpolation.
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
