# /// script
# dependencies = [
#   "paddlex>=3.0",
#   "paddlepaddle>=3.0",
#   "pypdfium2",
#   "opencv-contrib-python",
#   "shapely",
# ]
# ///

"""Reference layout detection output from PaddlePaddle's PP-DocLayoutV3.

Reads an image path via `--image`, runs PaddleX layout detection, and writes
a JSON array of blocks to stdout. Each block has the shape:

    {"label": str, "confidence": float, "bbox": [x, y, w, h], "order": int}

`bbox` is axis-aligned (x, y, width, height) derived from PaddleX's
`coordinate` (x1, y1, x2, y2). Used by the Rust parity tests in
`src-crates/core/tests/layout_parity.rs`.
"""

import argparse
import json
import sys

from paddlex import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = create_model("PP-DocLayoutV3")
    page = next(model.predict(args.image))

    blocks = []
    for box in page.get("boxes") or []:
        coordinate = box.get("coordinate")
        if not coordinate or len(coordinate) != 4:
            continue
        x1, y1, x2, y2 = (float(c) for c in coordinate)
        blocks.append(
            {
                "label": str(box.get("label", "")),
                "confidence": float(box.get("score", 0.0)),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "order": int(box.get("order", 0)),
            }
        )

    json.dump(blocks, sys.stdout)


if __name__ == "__main__":
    main()
