"""
Document structure parser — extracts headings, sections, paragraphs, tables,
and lists from PDFs using PyMuPDF font analysis.

No external dependency beyond PyMuPDF (already installed).
This gives us structure-aware chunking instead of blind fixed-size splitting.

Strategy:
  1. Analyse font sizes per page to calibrate heading thresholds.
  2. Walk every text block on every page.
  3. Classify each block as heading / paragraph / table / list.
  4. Group blocks into logical sections anchored by headings.
"""
import io
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz  # PyMuPDF (fitz alias)
from PIL import Image

from app.core.config import settings
from app.ingestion.metadata import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ParsedBlock:
    """A single classified content block within a page."""
    text: str
    block_type: str      # "heading" | "paragraph" | "table" | "list"
    page_number: int
    heading_level: int = 0   # 0 = not a heading; 1/2/3 = H1/H2/H3
    font_size: float = 11.0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def is_heading(self) -> bool:
        return self.block_type == "heading"


@dataclass
class ParsedSection:
    """
    A logical document section — a heading plus all the content blocks
    that follow it (until the next same-or-higher-level heading).
    """
    heading: str
    heading_level: int           # 1 = major section, 2 = sub, 3 = sub-sub
    page_start: int
    page_end: int
    blocks: List[ParsedBlock] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Concatenated text of all blocks, ready for chunking."""
        parts = []
        if self.heading:
            parts.append(self.heading)
        for b in self.blocks:
            if b.text.strip():
                parts.append(b.text.strip())
        return "\n\n".join(parts)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


# ============================================================================
# FONT STATISTICS
# ============================================================================

def _collect_font_sizes(doc: fitz.Document) -> List[float]:
    """Walk all spans in the document and collect font sizes."""
    sizes: List[float] = []
    for page in doc:
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0.0)
                    if size > 4:  # ignore tiny artifacts
                        sizes.append(size)
    return sizes


def _compute_thresholds(sizes: List[float]) -> Dict[str, float]:
    """
    Derive heading detection thresholds from the document's own font statistics.
    Using relative thresholds avoids hard-coding for specific fonts/sizes.
    """
    if not sizes:
        return {"body": 11.0, "h3": 12.5, "h2": 13.5, "h1": 15.0}

    body = statistics.median(sizes)
    return {
        "body": body,
        "h3": body * 1.10,   # slightly larger + bold = H3
        "h2": body * 1.20,   # noticeably larger = H2
        "h1": body * 1.40,   # clearly a title/H1
    }


# ============================================================================
# BLOCK CLASSIFICATION
# ============================================================================

def _spans_to_text(spans: list) -> str:
    return " ".join(s.get("text", "") for s in spans).strip()


def _is_bold(spans: list) -> bool:
    """Return True if any span in the block is bold."""
    for span in spans:
        flags = span.get("flags", 0)
        font = span.get("font", "").lower()
        if (flags & 16) or "bold" in font or "black" in font:
            return True
    return False


def _classify_block(
    spans: list,
    thresholds: Dict[str, float],
) -> Tuple[str, int, float]:
    """
    Classify a list of spans into a block type.

    Returns:
        (block_type, heading_level, dominant_font_size)
    """
    if not spans:
        return "paragraph", 0, 11.0

    text = _spans_to_text(spans)
    if not text:
        return "paragraph", 0, 11.0

    sizes = [s.get("size", 11.0) for s in spans]
    dominant_size = max(sizes)
    bold = _is_bold(spans)

    # --- List detection (bullet points / numbered items) ---
    if re.match(r"^[\u2022\u2023\u25e6•\-\*\u2013]\s", text) or \
       re.match(r"^\d{1,2}[\.\)]\s", text):
        return "list", 0, dominant_size

    # --- Short lines that are likely just noise or page numbers ---
    if len(text) < 5:
        return "paragraph", 0, dominant_size

    # --- Heading detection using font size thresholds ---
    if dominant_size >= thresholds["h1"]:
        return "heading", 1, dominant_size
    if dominant_size >= thresholds["h2"] or (dominant_size >= thresholds["h3"] and bold and len(text) < 120):
        return "heading", 2, dominant_size
    if dominant_size >= thresholds["h3"] and bold and len(text) < 80:
        return "heading", 3, dominant_size

    return "paragraph", 0, dominant_size


# ============================================================================
# MAIN PARSER
# ============================================================================

def parse_pdf_structure(
    file_bytes: bytes,
    document_name: str,
    document_id: Optional[str] = None,
    ocr_engine=None,
) -> List[ParsedSection]:
    """
    Parse a PDF into logical sections using font-size-based heading detection.

    Each section contains:
    - A heading (may be empty for introductory content before any heading)
    - A list of classified blocks (paragraph, list, table, heading)
    - Start and end page numbers

    Falls back to OCR for pages with insufficient text.

    Args:
        file_bytes: Raw PDF bytes
        document_name: Human-readable document name
        document_id: Stable document ID
        ocr_engine: Initialized PaddleOCR (used for scanned pages)

    Returns:
        List of ParsedSection objects
    """
    import tempfile, os

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)

        # 1. Calibrate font thresholds across the whole document
        all_sizes = _collect_font_sizes(doc)
        thresholds = _compute_thresholds(all_sizes)
        logger.info(
            f"'{document_name}': font thresholds = "
            f"body={thresholds['body']:.1f}, "
            f"h3={thresholds['h3']:.1f}, "
            f"h2={thresholds['h2']:.1f}, "
            f"h1={thresholds['h1']:.1f}"
        )

        all_blocks: List[ParsedBlock] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_number = page_num + 1

            # --- Attempt direct text extraction ---
            raw_text = page.get_text()

            if len(raw_text.strip()) < settings.OCR_MIN_TEXT_LENGTH and ocr_engine:
                # Scanned page → OCR → treat as single paragraph block
                from app.ingestion.ocr import extract_text_from_image
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(settings.OCR_DPI_SCALE, settings.OCR_DPI_SCALE)
                )
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text, method, conf = extract_text_from_image(ocr_engine, img)

                if ocr_text.strip():
                    all_blocks.append(
                        ParsedBlock(
                            text=ocr_text,
                            block_type="paragraph",
                            page_number=page_number,
                        )
                    )
                continue

            # --- Structure-aware extraction via get_text("dict") ---
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            # Extract tables first (PyMuPDF 1.23+ table finder)
            table_bboxes = set()
            try:
                table_finder = page.find_tables()
                for table in table_finder.tables:
                    rows = table.extract()
                    table_text = _table_to_text(rows)
                    if table_text.strip():
                        all_blocks.append(
                            ParsedBlock(
                                text=table_text,
                                block_type="table",
                                page_number=page_number,
                                bbox=tuple(table.bbox),
                            )
                        )
                    # Track bbox so we skip these blocks in text extraction
                    table_bboxes.add(tuple(int(x) for x in table.bbox))
            except Exception:
                pass  # Older PyMuPDF or no tables → skip

            # Walk text blocks
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:   # skip image blocks
                    continue

                # Skip blocks that overlap with extracted tables
                block_bbox = tuple(int(x) for x in block.get("bbox", (0, 0, 0, 0)))
                if _overlaps_any(block_bbox, table_bboxes):
                    continue

                # Collect all spans in block
                spans = []
                for line in block.get("lines", []):
                    spans.extend(line.get("spans", []))

                block_text = _spans_to_text(spans).strip()
                if not block_text:
                    continue

                block_type, h_level, font_size = _classify_block(spans, thresholds)

                all_blocks.append(
                    ParsedBlock(
                        text=block_text,
                        block_type=block_type,
                        page_number=page_number,
                        heading_level=h_level,
                        font_size=font_size,
                        bbox=block.get("bbox", (0, 0, 0, 0)),
                    )
                )

        doc.close()

    finally:
        os.unlink(tmp_path)

    # 2. Group blocks into sections anchored by headings
    sections = _group_into_sections(all_blocks)
    logger.info(f"'{document_name}': parsed into {len(sections)} sections")
    return sections


# ============================================================================
# SECTION GROUPING
# ============================================================================

def _group_into_sections(blocks: List[ParsedBlock]) -> List[ParsedSection]:
    """
    Group a flat list of blocks into hierarchical sections.

    A new section starts whenever a heading block is encountered.
    Content before the first heading goes into an "Introduction" section.
    """
    sections: List[ParsedSection] = []
    current_section: Optional[ParsedSection] = None

    for block in blocks:
        if block.is_heading:
            # Save the previous section
            if current_section is not None:
                if current_section.blocks or current_section.heading:
                    sections.append(current_section)

            current_section = ParsedSection(
                heading=block.text,
                heading_level=block.heading_level,
                page_start=block.page_number,
                page_end=block.page_number,
            )
        else:
            # Ensure we always have an active section
            if current_section is None:
                current_section = ParsedSection(
                    heading="",
                    heading_level=1,
                    page_start=block.page_number,
                    page_end=block.page_number,
                )
            current_section.blocks.append(block)
            current_section.page_end = block.page_number

    # Don't forget the last section
    if current_section is not None:
        if current_section.blocks or current_section.heading:
            sections.append(current_section)

    # If no sections were found at all, treat whole doc as one section
    if not sections:
        all_text = "\n\n".join(b.text for b in blocks if b.text.strip())
        page_start = blocks[0].page_number if blocks else 1
        page_end = blocks[-1].page_number if blocks else 1
        sections.append(
            ParsedSection(
                heading="Document Content",
                heading_level=1,
                page_start=page_start,
                page_end=page_end,
                blocks=blocks,
            )
        )

    return sections


# ============================================================================
# UTILITIES
# ============================================================================

def _table_to_text(rows: list) -> str:
    """Convert a 2D table (list of lists) to plain text."""
    if not rows:
        return ""
    lines = []
    for row in rows:
        cells = [str(c or "").strip() for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _overlaps_any(bbox: tuple, table_bboxes: set, threshold: float = 0.5) -> bool:
    """Return True if bbox significantly overlaps any table bbox."""
    if not table_bboxes or not bbox:
        return False
    bx0, by0, bx1, by1 = bbox
    for tb in table_bboxes:
        tx0, ty0, tx1, ty1 = tb
        ix0 = max(bx0, tx0)
        iy0 = max(by0, ty0)
        ix1 = min(bx1, tx1)
        iy1 = min(by1, ty1)
        inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        block_area = max(1, (bx1 - bx0) * (by1 - by0))
        if inter_area / block_area > threshold:
            return True
    return False
