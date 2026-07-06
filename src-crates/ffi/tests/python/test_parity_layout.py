from pathlib import Path

import akuna_core
import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from paddlex import create_model


HF_REPO_TEST_CORPUS = "akunasoftware/test-corpus"
IMAGE_FIXTURE = "content/fixtures/text-hidpi.png"
IOU_THRESHOLD = np.float32(0.85)
UNMATCHED_TOLERANCE = 0.0


def fixture_path(name: str) -> Path:
    """Return a test corpus fixture path."""
    return Path(hf_hub_download(HF_REPO_TEST_CORPUS, name, repo_type="dataset"))


def reference_blocks(image: Path) -> list[tuple[str, np.ndarray]]:
    """Return expected layout blocks."""
    model = create_model("PP-DocLayoutV3")
    page = next(model.predict(str(image)))
    blocks = []
    for box in page.get("boxes") or []:
        coordinate = box.get("coordinate")
        if not coordinate or len(coordinate) != 4:
            continue
        x1, y1, x2, y2 = np.asarray(coordinate, dtype=np.float32)
        blocks.append(
            (
                str(box.get("label", "")),
                np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
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
    actual: list[tuple[str, np.ndarray]],
    expected: list[tuple[str, np.ndarray]],
) -> None:
    """Assert layout blocks match by label and IoU."""
    actual_labels = {label for label, _ in actual}
    expected_labels = {label for label, _ in expected}
    assert actual_labels == expected_labels

    unmatched_actual = 0
    consumed = [False] * len(expected)
    for actual_label, actual_bbox in actual:
        candidates = [
            (iou(actual_bbox, expected_bbox), index)
            for index, (expected_label, expected_bbox) in enumerate(expected)
            if not consumed[index] and actual_label == expected_label
        ]
        if not candidates:
            unmatched_actual += 1
            continue
        best_iou, index = max(candidates, key=lambda item: item[0])
        if best_iou >= IOU_THRESHOLD:
            consumed[index] = True
        else:
            unmatched_actual += 1

    unmatched_expected = sum(not item for item in consumed)
    max_unmatched = int(np.ceil(max(len(actual), len(expected)) * UNMATCHED_TOLERANCE))
    assert unmatched_actual <= max_unmatched
    assert unmatched_expected <= max_unmatched


@pytest.mark.asyncio
async def test_layout_matches_paddlex() -> None:
    """Layout detection matches reference blocks."""
    path = fixture_path(IMAGE_FIXTURE)
    detector = await akuna_core.load_layout_detector(None)
    page = detector.detect_path(str(path))
    actual = [
        (
            block.label,
            np.array(
                [block.bbox.x, block.bbox.y, block.bbox.width, block.bbox.height],
                dtype=np.float32,
            ),
        )
        for block in page.blocks
    ]
    assert_blocks_match(actual, reference_blocks(path))
