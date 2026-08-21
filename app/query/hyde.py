"""
HyDE — Hypothetical Document Embedding (optional, off by default).

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al. 2022)

How it works:
  1. Generate a HYPOTHETICAL answer paragraph (as if the answer exists in the doc)
  2. Embed the hypothetical answer (instead of the raw query)
  3. Use this embedding for dense vector search

Why it helps:
  - Queries and document passages live in different semantic spaces
  - A hypothetical answer paragraph is much closer to an actual document passage
  - Especially effective for factoid and technical questions

Trade-off:
  - Extra LLM call (~400-600ms on Groq, near-instant)
  - Extra embedding call (~50ms)
  - Can backfire for queries where the LLM hallucinates a wrong hypothetical

Config: settings.ENABLE_HYDE = False (opt-in per query or globally)
"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in the domain. Write a 2-3 sentence hypothetical document "
     "passage that would contain the answer to the question below. "
     "Write as if this passage comes from an academic paper or technical report. "
     "Do NOT answer the question directly — write a passage ABOUT the topic."),
    ("human", "Question: {query}"),
])


def generate_hypothetical_document(query: str, llm) -> str:
    """
    Generate a hypothetical answer passage for the given query.

    Args:
        query: The user's question
        llm:   Any LangChain LLM instance

    Returns:
        A hypothetical document passage string for embedding
    """
    chain = _HYDE_PROMPT | llm

    try:
        result = chain.invoke({"query": query})
        hypothetical = result.content.strip() if hasattr(result, "content") else str(result)
        logger.debug(f"HyDE generated: {hypothetical[:100]}...")
        return hypothetical
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}. Using original query.")
        return query


def hyde_retrieve(
    query: str,
    retriever,
    llm,
) -> List[Document]:
    """
    Run HyDE retrieval: generate hypothetical answer → retrieve by it.

    Note: This modifies the retrieval query but keeps the original query
    visible to the LLM for answer generation.

    Args:
        query:     Original user query
        retriever: Any LangChain BaseRetriever
        llm:       LLM for hypothetical document generation

    Returns:
        Retrieved documents (based on the hypothetical embedding)
    """
    hypothetical_doc = generate_hypothetical_document(query, llm)

    try:
        docs = retriever.invoke(hypothetical_doc)

        # Store debug info for UI
        try:
            import streamlit as st
            debug = st.session_state.get("last_retrieval_debug", {})
            debug["hyde_doc"] = hypothetical_doc[:200]
            st.session_state["last_retrieval_debug"] = debug
        except Exception:
            pass

        logger.info(f"HyDE retrieved {len(docs)} docs")
        return docs
    except Exception as e:
        logger.warning(f"HyDE retrieval failed: {e}. Falling back to direct retrieval.")
        return retriever.invoke(query)
