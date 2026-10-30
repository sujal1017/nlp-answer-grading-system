"""Turns a PDF into per-page text: direct extraction where reliable, OCR where needed."""
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import pdf_processing as pdfp
from .ocr import OCRUnavailable, get_engine


@dataclass
class PageResult:
    page_number: int
    kind: str
    method: str            # direct_text | <ocr engine> | none
    text: str = ""
    engine_info: str = ""
    confidence: Optional[float] = None
    warnings: list = field(default_factory=list)
    error: Optional[str] = None


def process_pdf_pages(
    data: bytes,
    ocr_engine: str = "tesseract",
    ocr_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    dpi: int = 200,
    max_pages: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    engine_override=None,
) -> list[PageResult]:
    """Process every page. A failure on one page is recorded on that page and never aborts
    the others, and is never replaced with fabricated text."""
    doc = pdfp.open_pdf(data, max_pages)
    results: list[PageResult] = []
    try:
        total = doc.page_count
        engine = engine_override
        engine_error = None
        for i in range(total):
            info = pdfp.classify_page(doc[i])
            res = PageResult(page_number=i + 1, kind=info.kind, method="none")
            try:
                if info.kind == "empty":
                    res.warnings.append("Page appears blank (no text and no significant image).")
                elif info.kind in ("text", "mixed"):
                    res.method = "direct_text"
                    res.engine_info = "PyMuPDF text layer"
                    res.text = pdfp.extract_page_text(doc, i)
                    if info.kind == "mixed":
                        res.warnings.append(
                            "Page has a text layer and a large embedded image; content inside the image "
                            "(e.g. handwriting) was not read. Verify against the page image."
                        )
                else:  # scanned
                    if engine is None and engine_error is None:
                        try:
                            engine = get_engine(ocr_engine, ocr_model_id)
                        except OCRUnavailable as exc:
                            engine_error = str(exc)
                    if engine_error:
                        raise OCRUnavailable(engine_error)
                    image = pdfp.render_page_array(doc, i, dpi)
                    rot = pdfp.detect_rotation(image)
                    if rot:
                        image = pdfp.rotate_image(image, rot)
                        res.warnings.append(f"Page was rotated {rot} degrees clockwise for OCR.")
                    elif rot is None:
                        res.warnings.append("Page orientation could not be determined; assumed upright.")
                    ocr_in = pdfp.preprocess_for_ocr(image) if ocr_engine == "tesseract" else image
                    out = engine.recognize(ocr_in)
                    res.method = out.engine
                    res.engine_info = out.engine
                    res.text = out.text.strip()
                    res.confidence = out.confidence
                    res.warnings.extend(out.warnings)
                    if not res.text:
                        res.warnings.append("OCR returned no text for a page that contains an image.")
            except Exception as exc:  # recorded, not hidden
                res.error = f"{type(exc).__name__}: {exc}"
            results.append(res)
            if progress:
                progress((i + 1) / total, f"Processed page {i + 1} of {total}")
    finally:
        doc.close()
    return results
