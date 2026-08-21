"""
PDF Loader — Phase 2 upgrade.

Now uses document_parser.py for structure-aware extraction.
Returns ParsedSection objects alongside legacy DocumentChunk objects.

Falls back to basic page-by-page extraction if structure parsing fails.
"""
import io
import logging
import os
import tempfile
import uuid
from typing import List, Optional, Tuple

import pymupdf as fitz  # PyMuPDF (fitz alias)
from PIL import Image

from app.core.config import settings
from app.ingestion.metadata import DocumentChunk

logger = logging.getLogger(__name__)


def process_pdf_bytes_structured(
    file_bytes: bytes,
    document_name: str,
    document_id: Optional[str] = None,
    ocr_engine=None,
):
    """
    Process a PDF using structure-aware parsing (Phase 2).

    Returns a list of ParsedSection objects from document_parser.
    Automatically falls back to basic page-by-page extraction on error.

    Args:
        file_bytes: Raw PDF bytes
        document_name: Human-readable filename
        document_id: Stable document ID (generated if not provided)
        ocr_engine: Initialized PaddleOCR engine (optional)

    Returns:
        (document_id, sections) where sections is List[ParsedSection]
        or (document_id, None) if parsing failed (caller should use fallback)
    """
    if document_id is None:
        document_id = str(uuid.uuid4())

    try:
        from app.ingestion.document_parser import parse_pdf_structure
        sections = parse_pdf_structure(
            file_bytes=file_bytes,
            document_name=document_name,
            document_id=document_id,
            ocr_engine=ocr_engine,
        )
        return document_id, sections
    except Exception as e:
        logger.warning(
            f"Structure parsing failed for '{document_name}': {e}. "
            "Falling back to basic page extraction."
        )
        return document_id, None


def process_pdf_bytes(
    file_bytes: bytes,
    document_name: str,
    document_id: Optional[str] = None,
    ocr_engine=None,
) -> List[DocumentChunk]:
    """
    Fallback: process a PDF page-by-page and return DocumentChunk objects.
    Used when structure parsing fails or for the basic chunking path.
    """
    if document_id is None:
        document_id = str(uuid.uuid4())

    from app.ingestion.ocr import extract_text_from_image

    chunks: List[DocumentChunk] = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        doc = fitz.open(tmp_path)
        logger.info(f"Fallback processing '{document_name}' — {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            extraction_method = "PyMuPDF"
            ocr_confidence = 1.0

            if len(text.strip()) < settings.OCR_MIN_TEXT_LENGTH and ocr_engine:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(settings.OCR_DPI_SCALE, settings.OCR_DPI_SCALE)
                )
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text, extraction_method, ocr_confidence = extract_text_from_image(
                    ocr_engine, img
                )

            chunks.append(
                DocumentChunk(
                    content=text,
                    document_name=document_name,
                    document_id=document_id,
                    page_number=page_num + 1,
                    extraction_method=extraction_method,
                    ocr_confidence=ocr_confidence,
                )
            )
        doc.close()
    finally:
        os.unlink(tmp_path)

    return chunks


def process_pdf_file(
    file_path: str,
    document_name: Optional[str] = None,
    document_id: Optional[str] = None,
    ocr_engine=None,
) -> List[DocumentChunk]:
    """Convenience wrapper: process a PDF from a file path."""
    if document_name is None:
        document_name = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return process_pdf_bytes(
        file_bytes=file_bytes,
        document_name=document_name,
        document_id=document_id,
        ocr_engine=ocr_engine,
    )
