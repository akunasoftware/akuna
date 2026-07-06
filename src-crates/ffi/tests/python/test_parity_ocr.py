from pathlib import Path

import akuna_core
import pytest
from huggingface_hub import hf_hub_download
from paddleocr import PaddleOCR


HF_REPO_TEST_CORPUS = "akunasoftware/test-corpus"
IMAGE_FIXTURE = "content/fixtures/text-hidpi.png"
PADDLE_OCR_PARAMS = {
    "text_det_limit_side_len": 64,
    "text_det_limit_type": "min",
    "text_det_thresh": 0.3,
    "text_det_box_thresh": 0.6,
    "text_det_unclip_ratio": 1.5,
    "text_recognition_batch_size": 1,
    "text_rec_score_thresh": 0.0,
}
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
        # Measured fixture-truth floor: 0.997103.
        0.997,
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


def fixture_path(name: str) -> Path:
    """Return a test corpus fixture path."""
    return Path(hf_hub_download(HF_REPO_TEST_CORPUS, name, repo_type="dataset"))


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
    actual = aggregated_text(
        [block.text for block in ocr.extract_path(str(path)).blocks]
    )
    reference = PaddleOCR(
        text_detection_model_name=ref_detection_model,
        text_recognition_model_name=ref_recognition_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        **PADDLE_OCR_PARAMS,
    )
    expected = aggregated_text(
        list(reference.predict(str(path)))[0].json["res"]["rec_texts"]
    )

    assert similarity(actual, expected) >= reference_threshold, (
        tier,
        actual,
        expected,
    )
    assert similarity(actual, normalise_text(EXPECTED_TEXT)) >= truth_threshold
