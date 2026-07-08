import akuna_core
import pytest
from magika import Magika
from paddleocr import PaddleOCR

from test_parity_ocr import (
    BLOCK_IOU_THRESHOLD,
    IMAGE_FIXTURE,
    PADDLE_BOX_THRESH,
    PADDLE_OCR_PARAMS,
    SCORE_TOLERANCE,
    fixture_path,
    iou,
    normalise_text,
    reference_blocks,
    similarity,
)


OCR_OPTIONS = akuna_core.OcrEngineOptions(
    detection_model=akuna_core.OcrDetectionModel.PP_OCR_V6_MEDIUM,
    recognition_model=akuna_core.OcrRecognitionModel.PP_OCR_V6_MEDIUM,
    cache_dir=None,
)


def options() -> akuna_core.ExtractionOptions:
    """Return the complete extraction configuration."""
    return akuna_core.ExtractionOptions(
        return_metadata=True,
        return_content=True,
        return_parts=True,
        ocr=OCR_OPTIONS,
    )


@pytest.mark.asyncio
async def test_extract_path_ocr_matches_paddleocr() -> None:
    """OCR extraction provenance matches the PaddleOCR reference."""
    path = fixture_path(IMAGE_FIXTURE)
    result = await akuna_core.extract_path(str(path), options())
    assert result.metadata is not None
    reference_file_type = Magika().identify_path(path)
    assert result.metadata.stem == "text-hidpi"
    assert result.metadata.extension == "png"
    assert result.metadata.label == reference_file_type.output.label
    assert result.metadata.mime_type == reference_file_type.output.mime_type
    assert result.metadata.description == reference_file_type.output.description
    assert result.metadata.is_text == reference_file_type.output.is_text
    assert result.metadata.confidence == pytest.approx(
        reference_file_type.score,
        abs=5e-4,
    )
    assert len(result.metadata.hash) == 64
    assert int(result.metadata.hash, 16) >= 0

    reference = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        **PADDLE_OCR_PARAMS,
        text_det_box_thresh=PADDLE_BOX_THRESH["medium"],
    )
    reference_result = list(reference.predict(str(path)))[0].json["res"]
    assert isinstance(reference_result, dict)
    expected = reference_blocks(reference_result)
    assert result.parts is not None
    assert len(result.parts) == len(expected)

    consumed = [False] * len(expected)
    for index, part in enumerate(result.parts):
        assert part.index == index
        assert part.kind == akuna_core.PartKind.TEXT
        assert part.text is not None
        assert part.provenance is not None
        assert part.provenance.confidence is not None
        assert part.provenance.page is None
        assert part.provenance.bbox is not None
        assert part.provenance.byte_range is None
        actual_bbox = [
            part.provenance.bbox.x,
            part.provenance.bbox.y,
            part.provenance.bbox.width,
            part.provenance.bbox.height,
        ]
        candidates = [
            (iou(actual_bbox, expected_bbox), expected_index)
            for expected_index, (_, expected_bbox, _) in enumerate(expected)
            if not consumed[expected_index]
        ]
        best_iou, expected_index = max(candidates, key=lambda item: item[0])
        assert best_iou >= BLOCK_IOU_THRESHOLD
        expected_text, _, expected_confidence = expected[expected_index]
        assert (
            similarity(normalise_text(part.text), normalise_text(expected_text)) >= 0.99
        )
        assert part.provenance.confidence == pytest.approx(
            expected_confidence,
            abs=SCORE_TOLERANCE,
        )
        consumed[expected_index] = True

    assert result.text == "\n\n".join(
        part.text for part in result.parts if part.text is not None
    )
    assert len(result.pipeline) == 2
    detection, recognition = result.pipeline
    assert detection.step == akuna_core.ExtractionPipelineStepKind.DETECTION
    assert detection.engine == "magika"
    assert detection.outputs == {"types": 1}
    assert recognition.step == akuna_core.ExtractionPipelineStepKind.RECOGNITION
    assert recognition.engine == "PpOcrV6Medium"
    assert recognition.outputs == {"texts": len(expected)}
