"""
Query Decomposition — break complex/compound questions into sub-questions.

Problem this solves:
  "What are the model's accuracy and training time, and how does it compare
  to the baseline?" is three questions in one. A single retrieval pass will
  likely miss at least one of them.

Strategy:
  1. Classify if the question is complex (compound / multi-part / comparative)
  2. Decompose into ≤4 focused sub-questions
  3. Retrieve independently for each sub-question
  4. Deduplicate results
  5. Optionally: synthesize a combined answer (used for sub-question mode)

The LLM decomposer adds ~400-600ms but drastically improves complex query coverage.
"""
import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# ---- Prompts ----------------------------------------------------------------

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Classify this query into exactly one category:\n"
     "- 'simple': A single direct question answerable by one passage\n"
     "- 'complex': Multiple questions, comparisons, or multi-step reasoning needed\n"
     "- 'multi': The same question phrased in a way that benefits from semantic expansion\n\n"
     "Respond with ONLY one word: simple, complex, or multi"),
    ("human", "{query}"),
])

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Break this complex question into simpler, focused sub-questions. "
     "Each sub-question should be independently answerable from a document. "
     "Return ONLY the sub-questions, one per line. Maximum 4 sub-questions. "
     "No numbering or bullet points."),
    ("human", "{query}"),
])

# ---- Query Classification ---------------------------------------------------

def classify_query(query: str, llm) -> str:
    """
    Classify a user query as 'simple', 'complex', or 'multi'.

    Returns:
        One of: "simple" | "complex" | "multi"
    """
    # Fast heuristics first (avoid LLM call for obvious cases)
    lower = query.lower()
    word_count = len(query.split())

    # Very short queries → simple
    if word_count <= 6:
        return "simple"

    # Common comparison / multi-part indicators → complex
    complex_indicators = [
        " and ", " or ", " also ", " compare", " difference", " versus", " vs",
        " how does", " what are the", " explain both", " list all",
        " what is the relationship", " both ", " multiple",
    ]
    if any(ind in lower for ind in complex_indicators) and word_count > 15:
        return "complex"

    # Let the LLM classify ambiguous cases
    try:
        chain = _CLASSIFY_PROMPT | llm
        result = chain.invoke({"query": query})
        classification = result.content.strip().lower()
        if classification not in ("simple", "complex", "multi"):
            classification = "simple"  # Safe default
        logger.debug(f"Query classified as '{classification}': {query[:60]}")
        return classification
    except Exception as e:
        logger.warning(f"Query classification failed: {e}. Defaulting to 'simple'.")
        return "simple"


# ---- Decomposition ----------------------------------------------------------

def decompose_query(query: str, llm) -> List[str]:
    """
    Decompose a complex query into focused sub-questions.

    Returns:
        List of sub-question strings (may include the original if decomp fails)
    """
    try:
        chain = _DECOMPOSE_PROMPT | llm
        result = chain.invoke({"query": query})
        raw = result.content if hasattr(result, "content") else str(result)

        sub_questions = [
            line.strip().lstrip("•-*0123456789.) ")
            for line in raw.strip().split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

        # Keep max 2 sub-questions for speed (was 4)
        sub_questions = sub_questions[:2]

        if not sub_questions:
            logger.warning("Decomposer returned no sub-questions; using original.")
            return [query]

        logger.debug(f"Decomposed into {len(sub_questions)} sub-questions")
        return sub_questions

    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}. Using original.")
        return [query]


# ---- Decomposed Retrieval ---------------------------------------------------

def decomposed_retrieve(
    query: str,
    retriever,
    llm,
) -> Tuple[List[Document], List[str]]:
    """
    Decompose a complex query and retrieve for each sub-question.

    Returns:
        (deduplicated_docs, sub_questions_used)
    """
    sub_questions = decompose_query(query, llm)

    seen_ids: set = set()
    all_docs: List[Document] = []

    for sub_q in sub_questions:
        try:
            docs = retriever.invoke(sub_q)
            for doc in docs:
                key = (
                    doc.metadata.get("chunk_id", "")
                    or doc.metadata.get("parent_chunk_id", "")
                    or hash(doc.page_content[:200])
                )
                if key not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(key)
        except Exception as e:
            logger.warning(f"Retrieval failed for sub-question '{sub_q[:50]}': {e}")

    logger.info(
        f"Decomposed retrieval: {len(sub_questions)} sub-questions → "
        f"{len(all_docs)} unique docs"
    )

    # Store debug info for UI
    try:
        import streamlit as st
        debug = st.session_state.get("last_retrieval_debug", {})
        debug["sub_questions"] = sub_questions
        st.session_state["last_retrieval_debug"] = debug
    except Exception:
        pass

    return all_docs, sub_questions
