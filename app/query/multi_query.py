"""
Multi-Query Retrieval — expand a single query into N paraphrases.

Problem this solves:
  A single query phrasing may miss relevant chunks that use different terminology.
  E.g., "What is the model's accuracy?" may miss chunks about "performance metrics"
  or "F1 score" or "test results".

Strategy:
  1. Generate N alternative phrasings using an LLM
  2. Retrieve for EACH phrasing
  3. Deduplicate by chunk_id
  4. Return unique documents ordered by first-occurrence relevance

The extra LLM calls add ~300-500ms latency but significantly improve recall.
"""
import logging
import re
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings

logger = logging.getLogger(__name__)

_MULTI_QUERY_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at reformulating search queries for document retrieval. "
     "Generate {n} different versions of the user's question that capture the same "
     "information need from different angles. Use varied vocabulary and phrasing. "
     "Return ONLY the queries, one per line. No numbering, no explanations."),
    ("human", "Question: {query}"),
])


def generate_query_variants(
    query: str,
    llm,
    n: int = None,
) -> List[str]:
    """
    Generate N alternative phrasings of the query.

    Args:
        query: Original user query
        llm:   Any LangChain LLM instance
        n:     Number of variants (defaults to settings.MULTI_QUERY_COUNT)

    Returns:
        List of query strings including the original
    """
    n = n or settings.MULTI_QUERY_COUNT

    chain = _MULTI_QUERY_PROMPT_TEMPLATE | llm

    try:
        result = chain.invoke({"query": query, "n": n})
        raw = result.content if hasattr(result, "content") else str(result)

        variants = [
            line.strip().lstrip("•-*0123456789.) ")
            for line in raw.strip().split("\n")
            if line.strip() and "?" in line
        ]
        variants = [v for v in variants if len(v) > 10][:n]

        if not variants:
            logger.warning("Multi-query LLM returned no usable variants; using original.")
            return [query]

        # Always include the original query
        all_queries = [query] + variants
        logger.debug(f"Multi-query: generated {len(all_queries)} queries")
        return all_queries

    except Exception as e:
        logger.warning(f"Multi-query generation failed: {e}. Using original only.")
        return [query]


def multi_query_retrieve(
    query: str,
    retriever,
    llm,
    n: int = None,
    top_k_per_query: Optional[int] = None,
) -> List[Document]:
    """
    Run multi-query retrieval: generate query variants, retrieve for each,
    deduplicate by chunk_id, return unique documents.

    Args:
        query:            Original user query
        retriever:        Any LangChain BaseRetriever
        llm:              LLM for query generation
        n:                Number of query variants
        top_k_per_query:  Docs to retrieve per query (uses retriever's default if None)

    Returns:
        Deduplicated list of relevant Documents
    """
    queries = generate_query_variants(query, llm, n)

    seen_ids: set = set()
    all_docs: List[Document] = []

    for q in queries:
        try:
            docs = retriever.invoke(q)
            for doc in docs:
                cid = doc.metadata.get("chunk_id", "") or doc.metadata.get("parent_chunk_id", "")
                # Use content hash as fallback dedup key
                fallback_key = hash(doc.page_content[:200])
                key = cid or fallback_key

                if key not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(key)
        except Exception as e:
            logger.warning(f"Retrieval failed for variant '{q[:50]}': {e}")

    logger.info(
        f"Multi-query: {len(queries)} queries → {len(all_docs)} unique docs"
    )

    # Store debug info for UI
    try:
        import streamlit as st
        debug = st.session_state.get("last_retrieval_debug", {})
        debug["multi_queries"] = queries
        st.session_state["last_retrieval_debug"] = debug
    except Exception:
        pass

    return all_docs
