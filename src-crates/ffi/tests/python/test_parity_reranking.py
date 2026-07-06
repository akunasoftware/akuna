from functools import lru_cache

import akuna_core
import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PAIRS = [
    ("Rust machine learning", "Burn is a deep learning framework for Rust"),
    ("Rust machine learning", "Bananas are yellow"),
]
QUERY = "Rust machine learning"
DOCUMENTS = [
    "Bananas are yellow",
    "Burn is a deep learning framework for Rust",
    "Cargo builds Rust projects",
]


@lru_cache(maxsize=1)
def reference_model() -> tuple[object, object]:
    """Return a reranking reference model."""
    model_name = "BAAI/bge-reranker-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def reference_scores(pairs: list[tuple[str, str]]) -> np.ndarray:
    """Return reference reranker scores."""
    tokenizer, model = reference_model()
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=model.config.max_position_embeddings - 2,
        )
        scores = model(**inputs).logits.squeeze(-1).tolist()
    if isinstance(scores, float):
        scores = [scores]
    return np.asarray(scores, dtype=np.float32)


@pytest.mark.asyncio
async def test_reranking_scores_match_transformers() -> None:
    """Reranker scores match reference output."""
    reranker = await akuna_core.load_text_reranker(None)
    actual = np.asarray(
        reranker.score_pairs(
            [
                akuna_core.TextPair(query=query, document=document)
                for query, document in PAIRS
            ],
            2,
        ),
        dtype=np.float32,
    )
    expected = reference_scores(PAIRS)

    # Floor measured vs FlagEmbedding: worst score delta 5.7e-6 across pairs.
    assert actual.shape == expected.shape
    assert np.all(np.abs(actual - expected) <= np.float32(5e-5)), (actual, expected)


@pytest.mark.asyncio
async def test_rerank_matches_transformers() -> None:
    """Rerank output matches reference ordering and scores."""
    reranker = await akuna_core.load_text_reranker(None)
    options = akuna_core.RerankOptions(top_k=2, normalize=True, batch_size=2)
    actual = reranker.rerank(QUERY, DOCUMENTS, options)

    raw_scores = reference_scores([(QUERY, document) for document in DOCUMENTS])
    scores = 1.0 / (1.0 + np.exp(-raw_scores))
    expected = sorted(
        enumerate(zip(DOCUMENTS, scores, strict=True)),
        key=lambda item: item[1][1],
        reverse=True,
    )[:2]

    assert [result.index for result in actual] == [index for index, _ in expected]
    assert [result.document for result in actual] == [
        document for _, (document, _) in expected
    ]
    assert np.all(
        np.abs(
            np.asarray([result.score for result in actual], dtype=np.float32)
            - np.asarray([score for _, (_, score) in expected], dtype=np.float32)
        )
        <= np.float32(5e-5)
    )
