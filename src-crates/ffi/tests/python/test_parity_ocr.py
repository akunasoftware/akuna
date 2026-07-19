import akuna_core
import cv2
import numpy as np
import pytest
from paddleocr import PaddleOCR

from _fixtures import fixture_path

IMAGE_FIXTURE = "content/fixtures/text-hidpi.png"
PADDLE_OCR_PARAMS = {
    "text_det_limit_side_len": 736,
    "text_det_limit_type": "min",
    "text_det_thresh": 0.2,
    "text_det_unclip_ratio": 1.4,
    "text_recognition_batch_size": 1,
    "text_rec_score_thresh": 0.0,
}
PADDLE_BOX_THRESH = {"tiny": 0.4, "small": 0.45, "medium": 0.45}
# Calibrated against PaddleOCR on the hosted fixture: 0.8455 minimum IoU and
# 0.034561 maximum recognition-score delta across supported model tiers.
BLOCK_IOU_THRESHOLD = np.float32(0.84)
SCORE_TOLERANCE = np.float32(0.035)
TIERS = {
    "tiny": (
        akuna_core.OcrDetectionModel.PP_OCR_V6_TINY,
        akuna_core.OcrRecognitionModel.PP_OCR_V6_TINY,
        "PP-OCRv6_tiny_det",
        "PP-OCRv6_tiny_rec",
        # Measured PaddleOCR floor after crop parity: 0.999275.
        0.999,
        # Measured fixture-truth floor: 0.996376.
        0.996,
    ),
    "small": (
        akuna_core.OcrDetectionModel.PP_OCR_V6_SMALL,
        akuna_core.OcrRecognitionModel.PP_OCR_V6_SMALL,
        "PP-OCRv6_small_det",
        "PP-OCRv6_small_rec",
        # Measured PaddleOCR floor after unclip parity: 1.0.
        0.999,
        # Measured fixture-truth floor: 0.996376.
        0.996,
    ),
    "medium": (
        akuna_core.OcrDetectionModel.PP_OCR_V6_MEDIUM,
        akuna_core.OcrRecognitionModel.PP_OCR_V6_MEDIUM,
        "PP-OCRv6_medium_det",
        "PP-OCRv6_medium_rec",
        # Measured PaddleOCR floor after unclip parity: 1.0.
        0.999,
        # Measured fixture-truth floor: 0.997103.
        0.997,
    ),
}

EXPECTED_TEXT = """
For triflers, too, are they who, by their activities, weary themselves
in life, and have no settled aim to which they may direct, once and
for all, their every desire and project.
Source: section 7
On Looking Inward
Seldom are any found unhappy from not observing what is in the
minds of others.
But such as observe not well the stirrings of their own souls must of
necessity be unhappy.
Source: section 8
On Mortality
The duration of man's life is but an instant; his substance is fleeting,
his senses dull; the structure of his body corruptible; the soul but a
vortex.
In fine, the life of the body is but a river, and the life of the soul a
misty dream.
Existence is a warfare, and a journey in a strange land; and the end
of fame is to be forgotten.
Source: section 17
On Philosophy
What then avails to guide us? One thing, and one alone-Philosophy.
And this consists in keeping the divinity within inviolate and intact;
victorious over pain and pleasure; free from temerity, free from false-
hood, free from hypocrisy; independent of what others do or fail to
Source: section 17
Book III
On the Present Moment
Remember also that every man lives only this present moment, which
is a fleeting instant: the rest of time is either spent or quite unknown.
Short is the time which each of us has to live, and small the corner
of the earth he has to live in.
Source: section 10
2
"""


def normalise_text(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(text.lower().split())


def aggregated_text(blocks: list[str]) -> str:
    """Join and normalize OCR text blocks."""
    return normalise_text(" ".join(blocks))


def levenshtein(left: str, right: str) -> int:
    """Return Levenshtein edit distance."""
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    current = [0] * (len(right) + 1)
    for index, left_char in enumerate(left):
        current[0] = index + 1
        for right_index, right_char in enumerate(right):
            cost = 0 if left_char == right_char else 1
            current[right_index + 1] = min(
                current[right_index] + 1,
                previous[right_index + 1] + 1,
                previous[right_index] + cost,
            )
        previous, current = current, previous
    return previous[len(right)]


def similarity(left: str, right: str) -> float:
    """Return normalized edit similarity."""
    if not left and not right:
        return 1.0
    return 1.0 - (levenshtein(left, right) / max(len(left), len(right)))


def iou(left: np.ndarray, right: np.ndarray) -> np.float32:
    """Return axis-aligned IoU for [x, y, width, height]."""
    left_x1, left_y1, left_x2, left_y2 = (
        left[0],
        left[1],
        left[0] + left[2],
        left[1] + left[3],
    )
    right_x1, right_y1, right_x2, right_y2 = (
        right[0],
        right[1],
        right[0] + right[2],
        right[1] + right[3],
    )
    inter_width = max(np.float32(0.0), min(left_x2, right_x2) - max(left_x1, right_x1))
    inter_height = max(np.float32(0.0), min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = inter_width * inter_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return np.float32(0.0) if union <= 0 else np.float32(intersection / union)


def reference_blocks(
    result: dict[str, object],
) -> list[tuple[str, np.ndarray, np.float32]]:
    """Return PaddleOCR text, rectangles, and recognition confidence."""
    texts = result["rec_texts"]
    scores = result["rec_scores"]
    boxes = result["rec_boxes"]
    assert isinstance(texts, list)
    assert isinstance(scores, list)
    assert isinstance(boxes, list)
    return [
        (
            str(text),
            np.array(
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                dtype=np.float32,
            ),
            np.float32(score),
        )
        for text, score, box in zip(texts, scores, boxes, strict=True)
    ]


def assert_page_matches(
    page: akuna_core.OcrPage,
    expected: list[tuple[str, np.ndarray, np.float32]],
    width: int,
    height: int,
) -> None:
    """Assert every OCR output field matches the PaddleOCR reference."""
    assert page.width == width
    assert page.height == height
    assert len(page.blocks) == len(expected)

    consumed = [False] * len(expected)
    score_deltas = []
    for block in page.blocks:
        assert block.kind == akuna_core.OcrBlockKind.TEXT
        assert block.confidence is not None
        actual_bbox = np.array(
            [block.bbox.x, block.bbox.y, block.bbox.width, block.bbox.height],
            dtype=np.float32,
        )
        candidates = [
            (iou(actual_bbox, expected_bbox), index)
            for index, (_, expected_bbox, _) in enumerate(expected)
            if not consumed[index]
        ]
        best_iou, index = max(candidates, key=lambda item: item[0])
        assert best_iou >= BLOCK_IOU_THRESHOLD
        expected_text, _, expected_score = expected[index]
        assert (
            similarity(normalise_text(block.text), normalise_text(expected_text))
            >= 0.99
        )
        score_deltas.append(abs(block.confidence - expected_score))
        consumed[index] = True

    assert max(score_deltas, default=np.float32(0.0)) <= SCORE_TOLERANCE


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", TIERS)
async def test_ocr_matches_paddleocr(tier: str) -> None:
    """OCR text matches PaddleOCR reference."""
    (
        detection_model,
        recognition_model,
        ref_detection_model,
        ref_recognition_model,
        reference_threshold,
        truth_threshold,
    ) = TIERS[tier]
    path = fixture_path(IMAGE_FIXTURE)
    ocr = await akuna_core.load_ocr_engine(
        akuna_core.OcrEngineOptions(
            detection_model=detection_model,
            recognition_model=recognition_model,
            cache_dir=None,
        )
    )
    reference = PaddleOCR(
        text_detection_model_name=ref_detection_model,
        text_recognition_model_name=ref_recognition_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        **PADDLE_OCR_PARAMS,
        text_det_box_thresh=PADDLE_BOX_THRESH[tier],
    )
    reference_result = list(reference.predict(str(path)))[0].json["res"]
    assert isinstance(reference_result, dict)
    expected_blocks = reference_blocks(reference_result)
    expected = aggregated_text([text for text, _, _ in expected_blocks])
    image = cv2.imread(str(path))
    assert image is not None
    height, width = image.shape[:2]
    assert ocr.pipeline() == akuna_core.OcrPipeline(
        detection_model=detection_model,
        recognition_model=recognition_model,
    )

    bytes_page = ocr.extract_bytes(path.read_bytes())
    path_page = ocr.extract_path(str(path))
    assert_page_matches(bytes_page, expected_blocks, width, height)
    assert_page_matches(path_page, expected_blocks, width, height)
    actual = aggregated_text([block.text for block in bytes_page.blocks])

    assert similarity(actual, expected) >= reference_threshold, (
        tier,
        actual,
        expected,
    )
    assert similarity(actual, normalise_text(EXPECTED_TEXT)) >= truth_threshold
