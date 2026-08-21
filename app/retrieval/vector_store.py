"""
Vector store module — Phase 2 upgrade.

Adds ParentChildRetriever:
  1. Dense vector search on small child chunks (high precision)
  2. Look up the parent chunk for each hit (rich context for LLM)

Why this matters:
  - Searching small chunks = precise retrieval of the right passage
  - Returning large parent chunks = LLM gets enough context to answer well
  - Without this, we face a tradeoff: small chunks = precise but context-poor;
    large chunks = context-rich but retrieval is imprecise.

Phase 3 will replace ChromaDB with persistent Qdrant.
The ParentChildRetriever interface stays identical.
"""
import logging
import uuid
from typing import Dict, List, Optional

from langchain_pinecone import PineconeVectorStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level embedding model cache
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the shared embedding model instance (lazy, module-level cache)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


# ============================================================================
# Phase 1 — Simple vectorstore retriever (kept for fallback)
# ============================================================================

def build_vectorstore_retriever(
    documents: List[Document],
    top_k: int = None,
) -> VectorStoreRetriever:
    """
    Embed documents and build an ephemeral ChromaDB index.
    Used as fallback if structure parsing fails.
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    embedding_model = get_embedding_model()

    # We'll use a dynamic namespace per batch so it acts like an ephemeral collection
    # but lives inside the persistent Pinecone index.
    namespace = f"pdf_batch_{uuid.uuid4().hex}"

    logger.info(f"Building Pinecone (simple) with {len(documents)} chunks in namespace {namespace}...")
    
    # Check if we have an API key
    import os
    if not os.environ.get("PINECONE_API_KEY"):
        logger.warning("PINECONE_API_KEY not set. Pinecone insertion will fail.")

    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        index_name=settings.PINECONE_INDEX_NAME,
        namespace=namespace,
    )
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# ============================================================================
# Phase 2 — Parent-Child retriever
# ============================================================================

class ParentChildRetriever(BaseRetriever):
    """
    Retrieves child chunks by vector similarity, then returns their
    parent chunks for richer LLM context.

    Flow:
      Query → embed → find top-K child chunks
           → collect unique parent_chunk_ids
           → return parent Documents (larger, section-level context)

    If a child has no parent (edge case), the child itself is returned.
    """

    child_vectorstore: PineconeVectorStore
    parent_store: Dict[str, Document]
    top_k: int = 5
    fetch_k: int = 20   # fetch more children to maximise parent coverage

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        # 1. Dense search on child chunks
        child_hits = self.child_vectorstore.similarity_search(query, k=self.fetch_k)

        # 2. Collect unique parent IDs, preserving relevance order
        seen_parents: Dict[str, float] = {}  # parent_id → rank
        orphan_children: List[Document] = []

        for rank, child in enumerate(child_hits):
            parent_id = child.metadata.get("parent_chunk_id", "")
            if parent_id and parent_id not in seen_parents:
                seen_parents[parent_id] = rank
            elif not parent_id:
                orphan_children.append(child)

        # 3. Build result list: parents first (by relevance rank), then orphans
        results: List[Document] = []
        for parent_id in sorted(seen_parents, key=seen_parents.get):
            parent_doc = self.parent_store.get(parent_id)
            if parent_doc and len(results) < self.top_k:
                results.append(parent_doc)

        # Fill remaining slots with orphan children
        for child in orphan_children:
            if len(results) >= self.top_k:
                break
            results.append(child)

        logger.debug(
            f"ParentChildRetriever: {len(child_hits)} child hits → "
            f"{len(results)} parent docs returned"
        )
        return results


def build_parent_child_retriever(
    parent_docs: List[Document],
    child_docs: List[Document],
    top_k: int = None,
) -> ParentChildRetriever:
    """
    Build a ParentChildRetriever from parent and child document lists.

    Args:
        parent_docs: Full-section documents (sent to LLM)
        child_docs:  Sub-section chunks (embedded for retrieval)
        top_k:       Number of parent docs to return per query

    Returns:
        ParentChildRetriever instance
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    embedding_model = get_embedding_model()

    # Build parent lookup dict: parent_chunk_id → Document
    parent_store: Dict[str, Document] = {}
    for doc in parent_docs:
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id:
            parent_store[chunk_id] = doc

    namespace = f"children_{uuid.uuid4().hex}"

    logger.info(
        f"Building ParentChild Pinecone index: {len(parent_docs)} parents, "
        f"{len(child_docs)} children in namespace '{namespace}'"
    )

    if not child_docs:
        raise ValueError("No child documents provided — cannot build vector index.")

    child_vectorstore = PineconeVectorStore.from_documents(
        documents=child_docs,
        embedding=embedding_model,
        index_name=settings.PINECONE_INDEX_NAME,
        namespace=namespace,
    )

    return ParentChildRetriever(
        child_vectorstore=child_vectorstore,
        parent_store=parent_store,
        top_k=top_k,
        fetch_k=min(top_k * 4, 40),
    )


# ============================================================================
# Phase 3 — Hybrid Retriever factory
# ============================================================================

def build_hybrid_retriever(
    parent_docs: List[Document],
    child_docs: List[Document],
    top_k: int = None,
    fetch_k: int = None,
    enable_reranking: bool = True,
):
    """
    Build the full Phase 3 HybridRetriever.

    Constructs:
      - Dense vector index (ChromaDB) from child_docs
      - BM25 index from child_docs
      - Parent store (dict) from parent_docs
      - HybridRetriever orchestrating Dense + BM25 → RRF → Reranker → Parent lookup

    Args:
        parent_docs:      Full-section documents (sent to LLM)
        child_docs:       Sub-section chunks (indexed for retrieval)
        top_k:            Final number of parent docs to return
        fetch_k:          Candidates to fetch per retrieval method before RRF
        enable_reranking: Whether to run the cross-encoder reranker

    Returns:
        HybridRetriever (LangChain BaseRetriever)
    """
    from app.retrieval.bm25 import BM25Retriever
    from app.retrieval.hybrid import HybridRetriever

    top_k  = top_k  or settings.RETRIEVAL_TOP_K
    fetch_k = fetch_k or settings.RETRIEVAL_FETCH_K

    if not child_docs:
        raise ValueError("No child documents — cannot build hybrid retriever.")

    embedding_model = get_embedding_model()

    # Build parent store
    parent_store: Dict[str, Document] = {}
    for doc in parent_docs:
        cid = doc.metadata.get("chunk_id", "")
        if cid:
            parent_store[cid] = doc

    # Build dense vector index on child chunks
    namespace = f"hybrid_children_{uuid.uuid4().hex}"
    logger.info(
        f"Building Hybrid Pinecone index: {len(parent_docs)} parents, "
        f"{len(child_docs)} children, namespace={namespace}, reranking={enable_reranking}"
    )
    
    child_vectorstore = PineconeVectorStore.from_documents(
        documents=child_docs,
        embedding=embedding_model,
        index_name=settings.PINECONE_INDEX_NAME,
        namespace=namespace,
    )

    # Build BM25 index on the same child chunks
    bm25_retriever = BM25Retriever(documents=child_docs)

    return HybridRetriever(
        child_vectorstore=child_vectorstore,
        bm25_retriever=bm25_retriever,
        parent_store=parent_store,
        top_k=top_k,
        fetch_k=fetch_k,
        enable_reranking=enable_reranking,
    )

