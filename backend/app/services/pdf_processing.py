"""PDF inspection, per-page classification, rendering and image preprocessing.

Page kinds:
  text     - reliable selectable text, no significant images
  scanned  - little/no selectable text, page is dominated by an image
  mixed    - selectable text plus significant image content
  empty    - nothing detectable
"""
from dataclasses import dataclass, asdict
from typing import Optional

import fitz  # PyMuPDF
import numpy as np

MIN_TEXT_CHARS = 25          # fewer characters than this => no reliable text layer
IMAGE_AREA_FRACTION = 0.25   # image covering this fraction of the page counts as significant


class PDFError(ValueError):
    """Invalid, corrupted, encrypted or oversized PDF. Message is safe to show to users."""


@dataclass
class PageInfo:
    page_number: int  # 1-based
    kind: str
    text_chars: int
    image_count: int
    image_area_fraction: float
    rotation: int
    width: float
    height: float

    def to_dict(self):
        return asdict(self)


def open_pdf(data: bytes, max_pages: Optional[int] = None) -> "fitz.Document":
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupted or not a PDF
        raise PDFError(f"could not open PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise PDFError("PDF is password protected")
    if doc.page_count == 0:
        doc.close()
        raise PDFError("PDF has no pages")
    if max_pages and doc.page_count > max_pages:
        count = doc.page_count
        doc.close()
        raise PDFError(f"PDF has {count} pages; the configured limit is {max_pages}")
    return doc


def classify_page(page: "fitz.Page") -> PageInfo:
    text = page.get_text("text") or ""
    text_chars = len(text.strip())
    area = max(page.rect.width * page.rect.height, 1.0)
    img_area = 0.0
    img_count = 0
    try:
        for info in page.get_image_info():
            bbox = fitz.Rect(info["bbox"]) & page.rect
            if not bbox.is_empty:
                img_area += bbox.width * bbox.height
                img_count += 1
    except Exception:
        pass
    frac = min(img_area / area, 1.0)
    has_text = text_chars >= MIN_TEXT_CHARS
    big_image = frac >= IMAGE_AREA_FRACTION
    if has_text and big_image:
        kind = "mixed"
    elif has_text:
        kind = "text"
    elif big_image:
        kind = "scanned"
    else:
        kind = "empty"
    return PageInfo(
        page_number=page.number + 1,
        kind=kind,
        text_chars=text_chars,
        image_count=img_count,
        image_area_fraction=round(frac, 3),
        rotation=page.rotation,
        width=round(page.rect.width, 1),
        height=round(page.rect.height, 1),
    )


def analyze_pdf(data: bytes, max_pages: Optional[int] = None) -> list[PageInfo]:
    doc = open_pdf(data, max_pages)
    try:
        return [classify_page(p) for p in doc]
    finally:
        doc.close()


def extract_page_text(doc: "fitz.Document", page_index: int) -> str:
    """Selectable text of one page, in reading order."""
    return (doc[page_index].get_text("text", sort=True) or "").strip()


def render_page_png(doc: "fitz.Document", page_index: int, dpi: int = 200) -> bytes:
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    return pix.tobytes("png")


def render_page_array(doc: "fitz.Document", page_index: int, dpi: int = 200) -> np.ndarray:
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3].copy() if pix.n >= 3 else np.repeat(arr, 3, axis=2)


def detect_rotation(image: np.ndarray) -> Optional[int]:
    """Estimate the clockwise rotation (0/90/180/270) needed to make text upright.

    Uses Tesseract's orientation detection. Returns None when it cannot decide
    (e.g. blank page or very little text) - callers must not guess in that case.
    """
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        if float(osd.get("orientation_conf", 0)) < 1.5:
            return None
        return int(osd.get("rotate", 0)) % 360
    except Exception:
        return None


def rotate_image(image: np.ndarray, degrees_clockwise: int) -> np.ndarray:
    k = (degrees_clockwise // 90) % 4
    return np.ascontiguousarray(np.rot90(image, -k)) if k else image


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Gentle preprocessing intended for classical OCR (Tesseract).

    Grayscale + local contrast equalisation only. No binarisation or aggressive
    denoising, which can erase thin pen strokes. Vision-language models should be
    given the un-preprocessed colour render instead.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
