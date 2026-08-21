"""
Reciprocal Rank Fusion (RRF) — merge multiple ranked result lists.

RRF is the standard method for combining results from heterogeneous
retrieval systems (dense + sparse) without needing to normalise scores.

Original paper: Cormack et al., "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009).

Formula:
    RRF_score(d) = Σ [ 1 / (k + rank(d, list_i)) ]
    where k=60 is the smoothing constant from the original paper.
"""
import logging
from typing import Dict, List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# RRF smoothing constant — 60 is optimal per the original paper
_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = _RRF_K,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each element is a ranked list of (doc_id, original_score).
                      The doc_id is used to identify the same document across lists.
                      Original scores are NOT used — only rank positions matter.
        k:            RRF smoothing constant (default 60).

    Returns:
        Merged list of (doc_id, rrf_score) sorted descending by RRF score.
        Higher score = more relevant.
    """
    scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _original_score) in enumerate(ranked_list):
            if not doc_id:
                continue
            if doc_id not in scores:
                scores[doc_id] = 0.0
            # Add contribution from this ranked list
            scores[doc_id] += 1.0 / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"RRF merged {sum(len(r) for r in ranked_lists)} total hits → {len(merged)} unique")
    return merged


def apply_rrf_to_documents(
    dense_hits: List[Tuple[Document, float]],
    bm25_hits: List[Tuple[Document, float]],
    top_n: int = 40,
    k: int = _RRF_K,
) -> List[Tuple[Document, float]]:
    """
    High-level RRF helper: takes raw (Document, score) lists from dense and BM25,
    applies RRF, and returns merged (Document, rrf_score) list.

    Args:
        dense_hits:  Results from dense vector search — (Document, cosine_score)
        bm25_hits:   Results from BM25 search — (Document, bm25_score)
        top_n:       Max candidates to return after merging
        k:           RRF smoothing constant

    Returns:
        List of (Document, rrf_score) sorted descending, up to top_n items.
    """
    # Build ranked lists using chunk_id as the document identifier
    def to_ranked_list(hits: List[Tuple[Document, float]]) -> List[Tuple[str, float]]:
        ranked = []
        for i, (doc, score) in enumerate(hits):
            doc_id = doc.metadata.get("chunk_id", f"__idx_{i}__")
            ranked.append((doc_id, score))
        return ranked

    dense_ranked = to_ranked_list(dense_hits)
    bm25_ranked  = to_ranked_list(bm25_hits)

    rrf_scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked], k=k)

    # Build a lookup map: chunk_id → Document
    doc_map: Dict[str, Document] = {}
    for doc, _ in dense_hits:
        cid = doc.metadata.get("chunk_id", "")
        if cid:
            doc_map[cid] = doc
    for doc, _ in bm25_hits:
        cid = doc.metadata.get("chunk_id", "")
        if cid:
            doc_map[cid] = doc

    # Reconstruct ordered result list
    results: List[Tuple[Document, float]] = []
    for doc_id, rrf_score in rrf_scores[:top_n]:
        doc = doc_map.get(doc_id)
        if doc is not None:
            results.append((doc, rrf_score))

    return results
