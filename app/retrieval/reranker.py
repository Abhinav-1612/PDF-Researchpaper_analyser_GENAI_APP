"""
Cross-encoder reranker — precision re-scoring of retrieval candidates.

After BM25 + Dense → RRF gives us ~30-40 candidate chunks,
the cross-encoder scores each (query, chunk) pair jointly
(unlike bi-encoders which score them independently).

This joint scoring captures fine-grained relevance and significantly
improves precision, especially for nuanced queries.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fast (~180ms for 30 pairs on CPU)
  - Excellent quality for English documents
  - ~90MB download (cached after first run)
"""
import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level cross-encoder cache (loaded once per process)
_cross_encoder = None

# Maximum characters of chunk content to pass to the cross-encoder.
# Longer = more accurate but slower. 512 chars ≈ 100 tokens, fast on CPU.
_MAX_CHUNK_CHARS = 512


def get_cross_encoder():
    """
    Lazily load and cache the CrossEncoder model.
    First call triggers a model download (~90MB, cached in HuggingFace cache).
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        model_name = settings.RERANKER_MODEL
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        _cross_encoder = CrossEncoder(model_name, max_length=512)
        logger.info("Cross-encoder loaded successfully")
    return _cross_encoder


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Tuple[Document, float]]:
    """
    Rerank a list of candidate documents using cross-encoder scoring.

    Args:
        query:     The user's query (or rewritten query)
        documents: Candidate documents from RRF (30-50 items)
        top_k:     Number of results to return (defaults to settings.RERANKER_TOP_K)

    Returns:
        List of (Document, reranker_score) sorted descending.
        Scores are raw cross-encoder logits (higher = more relevant).
    """
    top_k = top_k or settings.RERANKER_TOP_K

    if not documents:
        return []

    model = get_cross_encoder()

    # Build (query, passage) pairs — truncate passages to control latency
    pairs = [
        (query, doc.page_content[:_MAX_CHUNK_CHARS])
        for doc in documents
    ]

    # Cross-encoder inference — single batched forward pass
    scores = model.predict(pairs, show_progress_bar=False)

    # Sort by score descending and return top-k
    scored: List[Tuple[Document, float]] = list(zip(documents, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    result = scored[:top_k]

    if result:
        logger.debug(
            f"Reranker: {len(documents)} candidates → {len(result)} results "
            f"(top score: {result[0][1]:.3f}, lowest: {result[-1][1]:.3f})"
        )

    return result
