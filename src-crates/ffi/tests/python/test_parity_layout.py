from pathlib import Path

import akuna_core
import cv2
import numpy as np
import pytest
from paddlex import create_model

from _fixtures import fixture_path

IMAGE_FIXTURE = "content/fixtures/text-hidpi.png"
IOU_THRESHOLD = np.float32(0.85)
# Calibrated from PP-DocLayoutV3 parity on the hosted fixture: 0.003990 max.
SCORE_TOLERANCE = np.float32(0.005)


def reference_blocks(image: Path) -> list[tuple[str, np.ndarray, np.float32, int]]:
    """Return expected layout blocks."""
    model = create_model("PP-DocLayoutV3")
    page = next(model.predict(str(image)))
    blocks = []
    for box in page.json["res"]["boxes"]:
        coordinate = box.get("coordinate")
        if not coordinate or len(coordinate) != 4:
            continue
        x1, y1, x2, y2 = np.asarray(coordinate, dtype=np.float32)
        blocks.append(
            (
                str(box.get("label", "")),
                np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
                np.float32(box["score"]),
                int(box["order"]),
            )
        )
    return blocks


def iou(a: np.ndarray, b: np.ndarray) -> np.float32:
    """Return axis-aligned IoU for [x, y, width, height]."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    inter_w = max(np.float32(0.0), min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(np.float32(0.0), min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return np.float32(0.0) if union <= 0 else np.float32(inter / union)


def assert_blocks_match(
    actual: list[tuple[str, np.ndarray, np.float32, int]],
    expected: list[tuple[str, np.ndarray, np.float32, int]],
    *,
    match_order: bool,
) -> None:
    """Assert layout blocks match by label and IoU."""
    unmatched_actual = 0
    consumed = [False] * len(expected)
    score_deltas = []
    for actual_label, actual_bbox, actual_score, actual_order in actual:
        candidates = [
            (iou(actual_bbox, expected_bbox), index)
            for index, (expected_label, expected_bbox, _, _) in enumerate(expected)
            if not consumed[index] and actual_label == expected_label
        ]
        if not candidates:
            unmatched_actual += 1
            continue
        best_iou, index = max(candidates, key=lambda item: item[0])
        if best_iou >= IOU_THRESHOLD:
            _, _, expected_score, expected_order = expected[index]
            score_deltas.append(abs(actual_score - expected_score))
            if match_order:
                assert actual_order == expected_order
            consumed[index] = True
        else:
            unmatched_actual += 1

    assert not unmatched_actual
    assert all(consumed)
    assert max(score_deltas, default=np.float32(0.0)) <= SCORE_TOLERANCE


@pytest.mark.asyncio
async def test_layout_matches_paddlex() -> None:
    """Layout detection matches reference blocks."""
    path = fixture_path(IMAGE_FIXTURE)
    detector = await akuna_core.load_layout_detector(
        akuna_core.LayoutDetectorOptions(
            model=akuna_core.LayoutModel.PP_DOC_LAYOUT_V3,
            cache_dir=None,
        )
    )
    path_page = detector.detect_path(str(path))
    bytes_page = detector.detect_bytes(path.read_bytes())
    image = cv2.imread(str(path))
    assert image is not None
    height, width = image.shape[:2]

    for page in (path_page, bytes_page):
        assert page.width == width
        assert page.height == height

    actual = [
        (
            block.label,
            np.array(
                [block.bbox.x, block.bbox.y, block.bbox.width, block.bbox.height],
                dtype=np.float32,
            ),
            np.float32(block.confidence),
            block.order,
        )
        for block in path_page.blocks
    ]
    expected = reference_blocks(path)
    assert_blocks_match(actual, expected, match_order=False)
    assert_blocks_match(
        [
            (
                block.label,
                np.array(
                    [
                        block.bbox.x,
                        block.bbox.y,
                        block.bbox.width,
                        block.bbox.height,
                    ],
                    dtype=np.float32,
                ),
                np.float32(block.confidence),
                block.order,
            )
            for block in bytes_page.blocks
        ],
        actual,
        match_order=True,
    )
