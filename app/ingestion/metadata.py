"""
DocumentChunk — the canonical data model for a processed document chunk.

Every piece of text flowing through the pipeline (from ingestion to retrieval)
is represented as a DocumentChunk. This ensures consistent metadata throughout.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of text from an ingested document,
    with rich metadata for filtering, citation, and parent-child retrieval.
    """

    # --- Core content ---
    content: str

    # --- Document identity ---
    document_name: str
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_version: str = "1.0"

    # --- Location ---
    page_number: int = 0
    section: str = ""
    subsection: str = ""

    # --- Chunk identity ---
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_chunk_id: Optional[str] = None   # set when this is a child chunk

    # --- Content classification ---
    content_type: str = "text"  # "text" | "table" | "list" | "heading" | "code"

    # --- Extraction ---
    extraction_method: str = "PyMuPDF"
    ocr_confidence: float = 1.0

    # --- Timestamps ---
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_metadata(self) -> dict:
        """
        Serialise to a flat dict suitable for ChromaDB/Qdrant metadata payload.
        All values must be JSON-serializable primitives.
        """
        return {
            "document_name": self.document_name,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "page_number": self.page_number,
            "section": self.section,
            "subsection": self.subsection,
            "chunk_id": self.chunk_id,
            "parent_chunk_id": self.parent_chunk_id or "",
            "content_type": self.content_type,
            "extraction_method": self.extraction_method,
            "ocr_confidence": float(self.ocr_confidence),
            "ingested_at": self.ingested_at,
        }

    @classmethod
    def from_metadata(cls, content: str, metadata: dict) -> "DocumentChunk":
        """Reconstruct a DocumentChunk from stored vector DB metadata."""
        return cls(
            content=content,
            document_name=metadata.get("document_name", ""),
            document_id=metadata.get("document_id", ""),
            document_version=metadata.get("document_version", "1.0"),
            page_number=metadata.get("page_number", 0),
            section=metadata.get("section", ""),
            subsection=metadata.get("subsection", ""),
            chunk_id=metadata.get("chunk_id", ""),
            parent_chunk_id=metadata.get("parent_chunk_id") or None,
            content_type=metadata.get("content_type", "text"),
            extraction_method=metadata.get("extraction_method", "unknown"),
            ocr_confidence=float(metadata.get("ocr_confidence", 1.0)),
            ingested_at=metadata.get("ingested_at", ""),
        )

    def to_langchain_document(self):
        """Convert to a LangChain Document for compatibility with LangChain chains."""
        from langchain_core.documents import Document
        return Document(page_content=self.content, metadata=self.to_metadata())
