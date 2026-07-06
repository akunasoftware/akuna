from pathlib import Path

import akuna_core
import pytest


@pytest.mark.asyncio
async def test_extract_path_text_file(tmp_path: Path) -> None:
    """Text file extraction returns requested fields."""
    path = tmp_path / "sample.txt"
    path.write_text("knowledge is more than memory", encoding="utf-8")

    result = await akuna_core.extract_path(
        str(path),
        akuna_core.ExtractionOptions(
            return_metadata=True,
            return_content=True,
            return_parts=True,
        ),
    )

    assert result.metadata is not None
    assert result.metadata.label == "txt"
    assert result.metadata.mime_type == "text/plain"
    assert result.text == "knowledge is more than memory"
    assert result.parts is not None
    assert [part.text for part in result.parts] == ["knowledge is more than memory"]


@pytest.mark.asyncio
async def test_extract_bytes_text_file() -> None:
    """Byte extraction returns text without path metadata."""
    result = await akuna_core.extract_bytes(
        b"knowledge is more than memory",
        akuna_core.ExtractionOptions(
            return_metadata=True,
            return_content=True,
            return_parts=True,
        ),
    )

    assert result.metadata is not None
    assert result.metadata.stem is None
    assert result.metadata.extension is None
    assert result.metadata.label == "txt"
    assert result.text == "knowledge is more than memory"


@pytest.mark.asyncio
async def test_extract_path_pipeline_step_kind(tmp_path: Path) -> None:
    """Pipeline steps use enum roles."""
    path = tmp_path / "sample.py"
    path.write_text(
        "def one():\n    return 1\n\ndef two():\n    return 2\n",
        encoding="utf-8",
    )

    result = await akuna_core.extract_path(
        str(path),
        akuna_core.ExtractionOptions(
            return_metadata=True,
            return_content=True,
            return_parts=True,
        ),
    )

    parsing_steps = [
        step
        for step in result.pipeline
        if step.step == akuna_core.ExtractionPipelineStepKind.PARSING
    ]
    assert parsing_steps
    assert parsing_steps[0].outputs["parts"] > 1
