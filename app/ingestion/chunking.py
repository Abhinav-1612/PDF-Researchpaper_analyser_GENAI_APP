"""
Chunking module — Phase 2 upgrade.

Implements two strategies:
  A. chunk_documents()          — basic fixed-size (Phase 1 backward compat)
  B. create_parent_child_chunks() — structure-aware parent-child chunking (NEW)

Parent-child chunking:
  Parent = full section (~1500-2500 chars) → sent to LLM for context
  Child  = small precise sub-chunk (~300-500 chars) → searched for relevance

This gives precision (finding the right passage) + context (returning the
full section so the LLM has enough information to answer well).
"""
import logging
import uuid
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.ingestion.document_parser import ParsedSection
from app.ingestion.metadata import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1 — Backward-compatible fixed-size chunker
# ============================================================================

def chunk_documents(
    page_chunks: List[DocumentChunk],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Split page-level DocumentChunks into fixed-size LangChain Documents.

    Phase 1 strategy: RecursiveCharacterTextSplitter.
    Still used as a fallback if structure parsing fails.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_splits: List[Document] = []
    for page_chunk in page_chunks:
        if not page_chunk.content.strip():
            continue
        lc_doc = page_chunk.to_langchain_document()
        splits = splitter.split_documents([lc_doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_index"] = i
            split.metadata["total_chunks_on_page"] = len(splits)
            split.metadata["page"] = page_chunk.page_number
        all_splits.extend(splits)

    logger.info(f"Fixed-size chunked {len(page_chunks)} pages → {len(all_splits)} chunks")
    return all_splits


# ============================================================================
# PHASE 2 — Structure-aware parent-child chunker
# ============================================================================

def create_parent_child_chunks(
    sections: List[ParsedSection],
    document_name: str,
    document_id: str,
    parent_size: int = None,
    child_size: int = None,
    child_overlap: int = None,
) -> Tuple[List[Document], List[Document]]:
    """
    Create parent and child document chunks from parsed sections.

    Parent chunks:
      - One per section (or split if section is very long)
      - Carry the full section context for LLM generation
      - Size: up to PARENT_CHUNK_SIZE chars

    Child chunks:
      - Multiple per section (small, precise sub-chunks)
      - Used for dense vector search (high precision)
      - Each carries a parent_chunk_id linking back to the parent
      - Size: CHUNK_SIZE chars (same as basic chunker)

    Returns:
        (parent_docs, child_docs)
        parent_docs → stored in parent store, looked up by ID
        child_docs  → embedded in vector store, searched for relevance
    """
    parent_size = parent_size or settings.PARENT_CHUNK_SIZE
    child_size = child_size or settings.CHUNK_SIZE
    child_overlap = child_overlap or settings.CHUNK_OVERLAP

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_docs: List[Document] = []
    child_docs: List[Document] = []

    for section in sections:
        full_text = section.full_text
        if not full_text.strip():
            continue

        # --- Create parent chunk(s) for this section ---
        parent_base_meta = {
            "document_name": document_name,
            "document_id": document_id,
            "section": section.heading,
            "heading_level": section.heading_level,
            "page_number": section.page_start,
            "page": section.page_start,         # legacy key for UI
            "page_end": section.page_end,
            "content_type": "parent",
            "extraction_method": "PyMuPDF+StructureParser",
        }

        parent_raw = parent_splitter.split_text(full_text)

        for p_idx, p_text in enumerate(parent_raw):
            parent_chunk_id = str(uuid.uuid4())
            parent_meta = {
                **parent_base_meta,
                "chunk_id": parent_chunk_id,
                "parent_chunk_id": "",          # parents have no parent
                "chunk_index": p_idx,
            }
            parent_docs.append(
                Document(page_content=p_text, metadata=parent_meta)
            )

            # --- Create child chunks from this parent ---
            child_raw = child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(child_raw):
                if not c_text.strip():
                    continue
                child_meta = {
                    "document_name": document_name,
                    "document_id": document_id,
                    "section": section.heading,
                    "heading_level": section.heading_level,
                    "page_number": section.page_start,
                    "page": section.page_start,  # legacy key for UI
                    "content_type": "child",
                    "extraction_method": "PyMuPDF+StructureParser",
                    "chunk_id": str(uuid.uuid4()),
                    "parent_chunk_id": parent_chunk_id,  # link to parent
                    "chunk_index": c_idx,
                }
                child_docs.append(
                    Document(page_content=c_text, metadata=child_meta)
                )

    logger.info(
        f"'{document_name}': {len(sections)} sections → "
        f"{len(parent_docs)} parent chunks, {len(child_docs)} child chunks"
    )
    return parent_docs, child_docs
