"""
OCR engine — PaddleOCR (primary) with Tesseract as fallback.

This module extracts text from page images when direct PDF text extraction
yields insufficient content (e.g., scanned documents, image-heavy pages).

Extracted from app.py (original implementation preserved).
"""
import logging
import platform
from typing import Tuple

import pytesseract
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Tesseract path configuration (Windows only) ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def init_paddle_ocr():
    """
    Initialize and return the PaddleOCR engine.
    
    Called once and cached at the Streamlit layer (or app startup).
    Disables MKL-DNN to prevent PIR AttributeErrors on CPU.
    """
    import os
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_onednn"] = "0"

    from paddleocr import PaddleOCR
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    # PaddleOCR 3.x API: use device='cpu' instead of deprecated use_gpu=False
    return PaddleOCR(
        use_angle_cls=True,
        lang="en",
        device="cpu",
        enable_mkldnn=False,
    )


def extract_text_paddle(ocr_engine, img_np) -> Tuple[str, float]:
    """
    Run PaddleOCR on a numpy image array.

    Returns:
        (extracted_text, average_confidence)
    """
    result = ocr_engine.predict(img_np)  # PaddleOCR 3.x API
    paddle_text = ""
    total_conf = 0.0
    count_conf = 0

    if result and len(result) > 0:
        ocr_result = result[0]
        if hasattr(ocr_result, "rec_texts") and ocr_result.rec_texts:
            for text_item, conf in zip(ocr_result.rec_texts, ocr_result.rec_scores):
                paddle_text += text_item + "\n"
                total_conf += float(conf)
                count_conf += 1

    avg_conf = (total_conf / count_conf) if count_conf > 0 else 0.0
    return paddle_text, avg_conf


def extract_text_tesseract(img: Image.Image) -> str:
    """Run Tesseract OCR on a PIL Image. Returns extracted text."""
    try:
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.warning(f"Tesseract failed: {e}")
        return ""


def extract_text_from_image(
    ocr_engine,
    img: Image.Image,
    confidence_threshold: float = None,
) -> Tuple[str, str, float]:
    """
    Full OCR extraction pipeline for a single page image.

    Strategy:
    1. PaddleOCR (high-fidelity)
    2. If confidence < threshold → Tesseract fallback
    3. If Tesseract also fails → return PaddleOCR result or empty string

    Args:
        ocr_engine: Initialized PaddleOCR instance
        img: PIL Image of the page
        confidence_threshold: Minimum confidence to accept PaddleOCR result

    Returns:
        (extracted_text, extraction_method, ocr_confidence)
    """
    import numpy as np

    if confidence_threshold is None:
        confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD

    img_np = np.array(img)

    # Level 1: PaddleOCR
    paddle_text, avg_conf = extract_text_paddle(ocr_engine, img_np)

    if avg_conf >= confidence_threshold and paddle_text.strip():
        logger.debug(f"PaddleOCR succeeded (conf={avg_conf:.2f})")
        return paddle_text, f"PaddleOCR (Conf: {avg_conf:.2f})", avg_conf

    # Level 2: Tesseract fallback
    logger.debug(f"PaddleOCR confidence too low ({avg_conf:.2f}), trying Tesseract")
    tesseract_text = extract_text_tesseract(img)

    if tesseract_text.strip():
        return tesseract_text, "Tesseract OCR (Fallback)", avg_conf

    # Level 3: Best-effort — return PaddleOCR result even if low confidence
    result_text = paddle_text if paddle_text.strip() else "No text could be extracted."
    return result_text, "OCR Engine (Low Confidence)", avg_conf
