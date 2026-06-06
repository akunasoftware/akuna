# /// script
# dependencies = [
#   "paddleocr>=3.0",
#   "paddlepaddle>=3.0",
# ]
# ///

"""Reference OCR output from PaddlePaddle's PP-OCRv6 Python runtime.

Reads an image path via `--image` and an optional `--tier` (tiny|small|medium),
runs PaddleOCR with tier-matched models, and writes a JSON array of blocks to
stdout. Each block has the shape:

    {"text": str, "bbox": [x1, y1, x2, y2], "confidence": float}

`bbox` is axis-aligned (min-x, min-y, max-x, max-y) regardless of the polygon
returned by PaddleOCR. Used by the Rust parity tests in
`src-crates/core/src/ocr/mod.rs`.
"""

import argparse
import json
import sys

from paddleocr import PaddleOCR


TIERS = {
    "tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--tier",
        choices=tuple(TIERS.keys()),
        default="medium",
    )
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    det_name, rec_name = TIERS[args.tier]
    ocr = PaddleOCR(
        lang=args.lang,
        text_detection_model_name=det_name,
        text_recognition_model_name=rec_name,
    )
    result = ocr.predict(args.image)

    blocks = []
    for page in result or []:
        # paddleocr 3.x returns dict-like objects with `rec_texts`,
        # `dt_polys`, and `rec_scores` arrays.
        texts = page.get("rec_texts") or []
        polys = page.get("dt_polys") or []
        scores = page.get("rec_scores") or []
        for text, poly, score in zip(texts, polys, scores):
            if not text:
                continue
            xs = [float(point[0]) for point in poly]
            ys = [float(point[1]) for point in poly]
            blocks.append(
                {
                    "text": str(text),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": float(score),
                }
            )

    json.dump(blocks, sys.stdout)


if __name__ == "__main__":
    main()
