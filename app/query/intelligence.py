"""
Query Intelligence Orchestrator — routes queries to the right retrieval strategy.

This is the main Phase 4 entry point.
It sits between query rewriting and the hybrid retriever.

Decision flow:
  Query
    ├── HyDE enabled? → use hypothetical embedding
    └── Classify query type:
          ├── "simple"  → direct hybrid retrieval (fastest)
          ├── "multi"   → multi-query expansion + merge
          └── "complex" → decomposition + sub-question retrieval + merge

The result is always a List[Document] ready for the QA chain.
"""
import logging
import time
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings

logger = logging.getLogger(__name__)


class QueryIntelligenceRetriever(BaseRetriever):
    """
    A LangChain BaseRetriever that wraps any base retriever with
    smart query routing: decomposition, multi-query, and HyDE.

    Usage:
        qi = QueryIntelligenceRetriever(
            base_retriever=hybrid_retriever,
            llm=llm,
        )
        docs = qi.invoke("What are the main results and how do they compare?")
    """

    base_retriever: Any                 # HybridRetriever or any BaseRetriever
    llm: Any                            # ChatGroq or any LangChain LLM
    enable_multi_query: bool = True
    enable_decomposition: bool = True
    enable_hyde: bool = False           # Off by default (latency + hallucination risk)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------ #

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:

        t_start = time.time()
        strategy_used = "simple"

        # ── HyDE path (if enabled) ─────────────────────────────────── #
        if self.enable_hyde or settings.ENABLE_HYDE:
            from app.query.hyde import hyde_retrieve
            docs = hyde_retrieve(query, self.base_retriever, self.llm)
            strategy_used = "hyde"

        # ── Query classification → routing ─────────────────────────── #
        else:
            from app.query.decomposition import classify_query, decomposed_retrieve
            from app.query.multi_query import multi_query_retrieve

            query_type = classify_query(query, self.llm)

            if query_type == "complex" and self.enable_decomposition:
                docs, sub_qs = decomposed_retrieve(query, self.base_retriever, self.llm)
                strategy_used = f"decomposed({len(sub_qs)} sub-qs)"

            elif query_type == "multi" and self.enable_multi_query:
                docs = multi_query_retrieve(query, self.base_retriever, self.llm)
                strategy_used = "multi_query"

            else:
                # Simple: direct retrieval (fastest path)
                docs = self.base_retriever.invoke(query)
                strategy_used = "simple"

        elapsed_ms = round((time.time() - t_start) * 1000, 1)
        logger.info(
            f"QueryIntelligence: strategy='{strategy_used}', "
            f"docs={len(docs)}, time={elapsed_ms}ms"
        )

        # Store strategy info in session state for the UI debug panel
        try:
            import streamlit as st
            debug = st.session_state.get("last_retrieval_debug", {})
            debug["qi_strategy"] = strategy_used
            debug["timings"] = debug.get("timings", {})
            debug["timings"]["query_intel_ms"] = elapsed_ms
            st.session_state["last_retrieval_debug"] = debug
        except Exception:
            pass

        return docs
