"""REAL OCR tests: these run the actual Tesseract binary. They are skipped if it is missing.

Qwen2-VL is NOT tested here: it needs torch + model weights that are not available in this
environment. Nothing in this file claims Qwen2-VL works.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services import pdf_processing as pdfp
from app.services.ingest import process_pdf_pages
from app.services.ocr import OCRUnavailable, QwenVLEngine, TesseractEngine, get_engine

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
HW = Path(__file__).parent / "fixtures" / "handwriting"


def test_real_tesseract_reads_clear_printed_text(pdfs):
    img = np.array(pdfs.text_image(["Distributed systems share resources", "Transparency hides distribution"], font_size=44))
    out = TesseractEngine().recognize(pdfp.preprocess_for_ocr(img))
    low = out.text.lower()
    assert "distributed" in low and "transparency" in low
    assert out.confidence is not None and out.confidence > 60  # genuine engine-provided confidence


def test_real_tesseract_blank_image_reports_no_text():
    out = TesseractEngine().recognize(np.full((600, 800), 255, dtype=np.uint8))
    assert out.text.strip() == "" and any("No text" in w for w in out.warnings)


def test_real_orientation_detection_fixes_upside_down_page(pdfs):
    upside = np.array(pdfs.text_image(["Distributed systems share resources across many computers",
                                       "Transparency hides the distribution from users"] * 3,
                                      font_size=40, rotate=180))
    rot = pdfp.detect_rotation(upside)
    assert rot == 180
    fixed = pdfp.rotate_image(upside, rot)
    assert "distributed" in TesseractEngine().recognize(pdfp.preprocess_for_ocr(fixed)).text.lower()


def test_real_ingest_of_synthetic_scanned_pdf(pdfs):
    data = pdfs.image([pdfs.text_image(["Question one answer text here", "second line of the answer"], font_size=44)])
    res = process_pdf_pages(data)
    assert res[0].method == "tesseract" and "answer" in res[0].text.lower()
    assert any("printed text" in w for w in res[0].warnings)  # handwriting caveat is always attached


def test_real_handwriting_photo_is_never_presented_as_reliable():
    """Real photographed handwriting from the project's qwen-ocr branch.

    Tesseract is not a handwriting recogniser. We do NOT assert good accuracy; we assert that the
    pipeline completes, never invents a confidence, and always carries the review warning.
    """
    img = np.array(Image.open(HW / "answer.jpeg").convert("RGB"))
    out = TesseractEngine().recognize(pdfp.preprocess_for_ocr(img))
    assert isinstance(out.text, str)
    assert any("handwriting" in w.lower() for w in out.warnings)
    if out.confidence is not None:
        assert 0 <= out.confidence <= 100


def test_qwen_engine_fails_gracefully_without_dependencies():
    """Real behaviour test of the unavailable path: no torch/transformers/weights here."""
    engine = QwenVLEngine("Qwen/Qwen2-VL-2B-Instruct")
    with pytest.raises(OCRUnavailable):
        engine.recognize(np.zeros((50, 50, 3), dtype=np.uint8))


def test_unavailable_engine_page_error_is_recorded(pdfs):
    res = process_pdf_pages(pdfs.image([pdfs.text_image(["x"])]), ocr_engine="qwen2vl")
    assert res[0].text == "" and res[0].error and "OCRUnavailable" in res[0].error


def test_unknown_engine_name_rejected():
    with pytest.raises(OCRUnavailable):
        get_engine("no-such-engine")
