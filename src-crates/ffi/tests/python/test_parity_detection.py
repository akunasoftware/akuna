from pathlib import Path
from functools import lru_cache

import akuna_core
from huggingface_hub import hf_hub_download, list_repo_files
from magika import Magika
import pytest


HF_REPO_TEST_CORPUS = "akunasoftware/test-corpus"
HF_CONTENT_PREFIX = "content/"
SAMPLES = [
    ("sample.txt", b"knowledge is more than memory\n"),
    ("sample.pdf", b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"),
]


@lru_cache(maxsize=1)
def corpus_paths() -> tuple[tuple[str, Path], ...]:
    """Return downloaded test-corpus content files."""
    names = sorted(
        name
        for name in list_repo_files(HF_REPO_TEST_CORPUS, repo_type="dataset")
        if name.startswith(HF_CONTENT_PREFIX)
    )
    assert names
    return tuple(
        (
            name,
            Path(hf_hub_download(HF_REPO_TEST_CORPUS, name, repo_type="dataset")),
        )
        for name in names
    )


def assert_file_type_matches(actual: akuna_core.FileType, expected: object) -> None:
    """Assert file type metadata matches."""
    info = expected.output
    assert actual.info.label == info.label
    assert actual.info.mime_type == info.mime_type
    assert actual.info.group == info.group
    assert actual.info.description == info.description
    assert actual.info.extensions == info.extensions
    assert actual.info.is_text == info.is_text


def assert_model_result(actual: akuna_core.FileType, expected: object) -> None:
    """Assert model confidence and origin against the Magika reference."""
    assert actual.confidence == pytest.approx(expected.score, abs=5e-4)
    assert actual.origin == akuna_core.DetectionOrigin.MODEL


def test_identify_bytes_and_file_match(tmp_path: Path) -> None:
    """Toy file type detection matches reference output."""
    reference = Magika()
    detector = akuna_core.FileTypeDetector()

    for name, data in SAMPLES:
        path = tmp_path / name
        path.write_bytes(data)

        bytes_result = detector.identify_bytes(data)
        path_result = detector.identify_path(str(path))
        assert_file_type_matches(bytes_result, reference.identify_bytes(data))
        assert_file_type_matches(path_result, reference.identify_path(path))
        assert_model_result(bytes_result, reference.identify_bytes(data))
        assert_model_result(path_result, reference.identify_path(path))


def test_identify_corpus_matches_reference() -> None:
    """File type detection matches reference output across the corpus."""
    reference = Magika()
    detector = akuna_core.FileTypeDetector()

    for _name, path in corpus_paths():
        data = path.read_bytes()
        bytes_result = detector.identify_bytes(data)
        path_result = detector.identify_path(str(path))
        assert_file_type_matches(bytes_result, reference.identify_bytes(data))
        assert_file_type_matches(path_result, reference.identify_path(path))
        assert 0.0 <= bytes_result.confidence <= 1.0
        assert 0.0 <= path_result.confidence <= 1.0
