from functools import lru_cache

import akuna_core
import numpy as np
import pytest
from sentence_transformers import SentenceTransformer


BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

PARITY_TEXTS = [
    "Hello world",
    "Rust embeddings",
    "Semantic search: fast, accurate, and simple.",
    "  padded input with leading and trailing spaces  ",
    "Numbers 12345, symbols !?., and mixed CASE.",
    "emoji rocket and unicode cafe",
    "Machine learning in Rust.",
    "Apprendimento automatico in Rust.",
    "Rust での機械学習",
]

LONG_PARITY_TEXTS = [
    "Burn embeddings should match sentence-transformers even when tokenizer "
    "truncation is required. " * 128
]

MODEL_CASES = [
    (
        akuna_core.EmbeddingModel.BGE_BASE_EN_V15,
        "BAAI/bge-base-en-v1.5",
        "document",
        None,
    ),
    (
        akuna_core.EmbeddingModel.BGE_LARGE_EN_V15,
        "BAAI/bge-large-en-v1.5",
        "document",
        None,
    ),
    (
        akuna_core.EmbeddingModel.BGE_SMALL_EN_V15,
        "BAAI/bge-small-en-v1.5",
        "document",
        None,
    ),
    (
        akuna_core.EmbeddingModel.BGE_SMALL_EN_V15,
        "BAAI/bge-small-en-v1.5",
        "query",
        BGE_QUERY_PROMPT,
    ),
    (
        akuna_core.EmbeddingModel.MINI_LM_L12,
        "sentence-transformers/all-MiniLM-L12-v2",
        "document",
        None,
    ),
    (
        akuna_core.EmbeddingModel.MINI_LM_L6,
        "sentence-transformers/all-MiniLM-L6-v2",
        "document",
        None,
    ),
    (
        akuna_core.EmbeddingModel.ALL_MPNET_BASE_V2,
        "sentence-transformers/all-mpnet-base-v2",
        "document",
        None,
    ),
    (akuna_core.EmbeddingModel.BGE_M3, "BAAI/bge-m3", "document", None),
]


@lru_cache(maxsize=None)
def reference_model(name: str) -> SentenceTransformer:
    """Return an embedding reference model."""
    return SentenceTransformer(name)


def max_delta_tolerance(model: akuna_core.EmbeddingModel) -> float:
    """Return calibrated embedding component tolerance."""
    # Floors measured against sentence-transformers (worst component delta over
    # all parity texts/input kinds): the BERT-family models all land <= 6.1e-7,
    # while BgeM3 (24-layer XLM-R-large) accumulates ~30x more at 1.8e-5. These
    # bounds sit a small factor above the observed worst -- tight enough that any
    # real regression trips them, with headroom only for f32 op-ordering drift.
    return 1e-4 if model == akuna_core.EmbeddingModel.BGE_M3 else 5e-6


def reference_embeddings(
    model_name: str,
    texts: list[str],
    kind: str,
    prompt: str | None,
) -> np.ndarray:
    """Return reference embeddings."""
    model = reference_model(model_name)
    if prompt is None:
        embeddings = model.encode(texts, normalize_embeddings=True)
    elif kind == "query" and hasattr(model, "encode_query"):
        embeddings = model.encode_query(texts, prompt=prompt, normalize_embeddings=True)
    elif kind == "document" and hasattr(model, "encode_document"):
        embeddings = model.encode_document(
            texts, prompt=prompt, normalize_embeddings=True
        )
    else:
        embeddings = model.encode(texts, prompt=prompt, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def assert_embeddings_close(
    actual: list[list[float]],
    expected: np.ndarray,
    model: akuna_core.EmbeddingModel,
    kind: str,
    tolerance: float,
) -> None:
    """Assert embedding batches match tolerance."""
    actual_array = np.asarray(actual, dtype=np.float32)
    assert actual_array.shape == expected.shape
    max_deltas = np.max(np.abs(actual_array - expected), axis=1)
    assert np.all(max_deltas <= np.float32(tolerance)), (model, kind, max_deltas)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "reference", "kind", "prompt"), MODEL_CASES)
async def test_embedding_matches_sentence_transformers(
    model: akuna_core.EmbeddingModel,
    reference: str,
    kind: str,
    prompt: str | None,
) -> None:
    """Embedding output matches reference models."""
    embedder = await akuna_core.load_text_embedder(
        akuna_core.TextEmbedderOptions(model=model, cache_dir=None)
    )
    assert embedder.model() == model
    for texts in (PARITY_TEXTS, LONG_PARITY_TEXTS):
        expected = reference_embeddings(reference, texts, kind, prompt)
        if prompt is None:
            actual = embedder.embed_batch(texts, 2)
            single = embedder.embed(texts[0])
        else:
            actual = embedder.embed_batch_with_prompt(texts, 2, prompt)
            single = embedder.embed_with_prompt(texts[0], prompt)
        assert_embeddings_close(
            actual, expected, model, kind, max_delta_tolerance(model)
        )
        assert_embeddings_close(
            [single], expected[:1], model, kind, max_delta_tolerance(model)
        )
