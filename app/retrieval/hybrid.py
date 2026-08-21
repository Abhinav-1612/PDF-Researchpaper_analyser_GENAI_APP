"""
Hybrid Retriever — orchestrates the full Phase 3 retrieval pipeline.

Pipeline per query:
  1. Dense Search   → top FETCH_K child chunks (semantic similarity)
  2. BM25 Search    → top FETCH_K child chunks (keyword matching)
  3. RRF            → merge and re-rank → top 30-40 candidates
  4. Cross-Encoder  → precision re-score → top RERANKER_TOP_K chunks
  5. Parent Lookup  → return parent section documents to LLM

Why each step:
  Dense:      Handles paraphrase / semantic matches ("performance" ↔ "accuracy")
  BM25:       Handles exact term matches ("EfficientNet", "98.5%", "mIoU")
  RRF:        Combines rankings without score normalisation (proven effective)
  Reranker:   Joint query-passage scoring — much higher precision than bi-encoder
  Parent:     Large context window → LLM has the full section, not a 400-char fragment
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.rrf import apply_rrf_to_documents

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Full hybrid retrieval pipeline implementing:
    Dense + BM25 → RRF → Cross-Encoder → Parent-Child Lookup

    Designed as a drop-in replacement for the Phase 2 ParentChildRetriever —
    uses the same interface so nothing else in the codebase changes.
    """

    # Pydantic fields (LangChain BaseRetriever uses Pydantic v2)
    child_vectorstore: Any          # Chroma instance
    bm25_retriever: Any             # BM25Retriever instance
    parent_store: Dict[str, Document]
    top_k: int = 5
    fetch_k: int = 20
    enable_reranking: bool = True

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------ #
    # Core retrieval method (called by LangChain)
    # ------------------------------------------------------------------ #

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:

        debug: Dict[str, Any] = {"query": query, "timings": {}}
        t_total = time.time()

        # ── Step 1: Dense search ──────────────────────────────────────── #
        t0 = time.time()
        try:
            dense_hits: List[Tuple[Document, float]] = (
                self.child_vectorstore.similarity_search_with_score(query, k=self.fetch_k)
            )
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")
            dense_hits = []
            
        # Store dense scores in metadata
        for doc, score in dense_hits:
            doc.metadata["dense_score"] = round(float(score), 4)
            
        debug["timings"]["dense_ms"] = round((time.time() - t0) * 1000, 1)
        debug["dense_hits"] = len(dense_hits)

        # ── Step 2: BM25 search ───────────────────────────────────────── #
        t0 = time.time()
        try:
            bm25_hits: List[Tuple[Document, float]] = self.bm25_retriever.retrieve(
                query, top_k=self.fetch_k
            )
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            bm25_hits = []
            
        # Store BM25 scores in metadata
        for doc, score in bm25_hits:
            doc.metadata["bm25_score"] = round(float(score), 4)
            
        debug["timings"]["bm25_ms"] = round((time.time() - t0) * 1000, 1)
        debug["bm25_hits"] = len(bm25_hits)

        # ── Step 3: RRF merge ─────────────────────────────────────────── #
        t0 = time.time()
        rrf_results: List[Tuple[Document, float]] = apply_rrf_to_documents(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            top_n=self.fetch_k,
        )
        # If one method returned nothing, fall back to the other
        if not rrf_results:
            rrf_results = dense_hits or bm25_hits
            
        # Store RRF scores in metadata
        for doc, score in rrf_results:
            doc.metadata["rrf_score"] = round(float(score), 4)
            
        candidates: List[Document] = [doc for doc, _ in rrf_results]
        debug["timings"]["rrf_ms"] = round((time.time() - t0) * 1000, 1)
        debug["rrf_candidates"] = len(candidates)

        # ── Step 4: Cross-encoder reranking ───────────────────────────── #
        if self.enable_reranking and candidates:
            t0 = time.time()
            try:
                from app.retrieval.reranker import rerank_documents
                reranked = rerank_documents(query, candidates, top_k=self.top_k * 2)
                
                # Store reranker scores
                for doc, score in reranked:
                    doc.metadata["rerank_score"] = round(float(score), 4)
                    
                candidates = [doc for doc, score in reranked]
                debug["reranker_scores"] = [round(s, 3) for _, s in reranked[:5]]
            except Exception as e:
                logger.warning(f"Reranking failed (using RRF order): {e}")
            debug["timings"]["rerank_ms"] = round((time.time() - t0) * 1000, 1)
        debug["after_rerank"] = len(candidates)

        # ── Step 5: Parent-child lookup with per-document fairness ────── #
        # Resolve all candidates to their parent docs first.
        # Use a dict keyed by parent_id so we accumulate the BEST score
        # across ALL children that map to the same parent.
        parent_best: Dict[str, Dict] = {}   # parent_id → {doc, scores}
        standalone: List[Document] = []
        seen_child_ids: set = set()

        for child in candidates:
            parent_id = child.metadata.get("parent_chunk_id", "")
            child_id  = child.metadata.get("chunk_id", "")

            if parent_id:
                parent_doc = self.parent_store.get(parent_id)
                if parent_doc:
                    if parent_id not in parent_best:
                        parent_best[parent_id] = {
                            "doc": parent_doc,
                            "dense_score":  None,
                            "bm25_score":   None,
                            "rrf_score":    None,
                            "rerank_score": None,
                        }
                    # Keep the best (highest) non-None score for each metric
                    entry = parent_best[parent_id]
                    for metric in ("dense_score", "bm25_score", "rrf_score", "rerank_score"):
                        child_val = child.metadata.get(metric)
                        if child_val is not None:
                            if entry[metric] is None or child_val > entry[metric]:
                                entry[metric] = child_val

            elif child_id not in seen_child_ids:
                standalone.append(child)
                seen_child_ids.add(child_id)

        # Stamp the best accumulated scores onto each parent document
        resolved: List[Document] = []
        for parent_id, entry in parent_best.items():
            doc = entry["doc"]
            doc.metadata["dense_score"]  = entry["dense_score"]
            doc.metadata["bm25_score"]   = entry["bm25_score"]
            doc.metadata["rrf_score"]    = entry["rrf_score"]
            doc.metadata["rerank_score"] = entry["rerank_score"]
            resolved.append(doc)
        resolved.extend(standalone)

        # ── Fairness pass: guarantee ≥2 slots per unique document ─────── #
        # This ensures multi-PDF questions (e.g. "compare both") get
        # content from EVERY uploaded document, not just the top-scoring one.
        unique_docs = list(dict.fromkeys(
            d.metadata.get("document_name", "") for d in resolved
        ))
        results: List[Document] = []
        MIN_PER_DOC = 2  # guaranteed minimum slots per document

        if len(unique_docs) > 1:
            # Phase A: Fill guaranteed slots for each document
            per_doc_counts: Dict[str, int] = {name: 0 for name in unique_docs}
            remaining: List[Document] = []
            for doc in resolved:
                doc_name = doc.metadata.get("document_name", "")
                if per_doc_counts.get(doc_name, 0) < MIN_PER_DOC:
                    results.append(doc)
                    per_doc_counts[doc_name] = per_doc_counts.get(doc_name, 0) + 1
                else:
                    remaining.append(doc)
            # Phase B: Fill leftover slots by original rerank order
            for doc in remaining:
                if len(results) >= self.top_k:
                    break
                results.append(doc)
        else:
            # Single document — just take top_k by rank
            results = resolved[:self.top_k]

        debug["final_results"] = len(results)
        debug["timings"]["total_ms"] = round((time.time() - t_total) * 1000, 1)

        # Store debug info for the UI panel (best-effort — non-critical)
        _store_debug(debug)

        logger.info(
            f"HybridRetrieval: dense={debug['dense_hits']}, "
            f"bm25={debug['bm25_hits']}, "
            f"rrf={debug['rrf_candidates']}, "
            f"reranked={debug['after_rerank']}, "
            f"final={debug['final_results']}, "
            f"total={debug['timings']['total_ms']}ms"
        )
        return results


def _store_debug(debug: Dict[str, Any]) -> None:
    """Store retrieval debug data (No-op after removing Streamlit)."""
    pass
