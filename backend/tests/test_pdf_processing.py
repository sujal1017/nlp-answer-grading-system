"""PDF inspection and ingestion. All PDFs are synthetic fixtures built in conftest.py."""
import pytest

from app.services import pdf_processing as pdfp
from app.services.ingest import process_pdf_pages
from app.services.ocr import OCRResult


class FakeOCR:
    """MOCK engine used only to isolate ingestion logic from real OCR."""
    name = "fake"

    def __init__(self, text="mock ocr text"):
        self.text, self.calls = text, 0

    def recognize(self, image):
        self.calls += 1
        return OCRResult(text=self.text, engine="fake")


def test_text_pdf_is_classified_text(pdfs):
    infos = pdfp.analyze_pdf(pdfs.text(["Hello world this is a page with enough selectable text to count."]))
    assert [i.kind for i in infos] == ["text"]


def test_scanned_pdf_is_classified_scanned(pdfs):
    infos = pdfp.analyze_pdf(pdfs.image([pdfs.text_image(["scanned page"])]))
    assert infos[0].kind == "scanned" and infos[0].text_chars == 0


def test_mixed_pdf_is_classified_mixed(pdfs):
    assert pdfp.analyze_pdf(pdfs.mixed())[0].kind == "mixed"


def test_blank_page_is_empty(pdfs):
    assert pdfp.analyze_pdf(pdfs.blank(2))[1].kind == "empty"


def test_multipage_mixed_kinds(pdfs):
    data = pdfs.text(["First page has plenty of selectable text on it, enough to be reliable."])
    infos = pdfp.analyze_pdf(data)
    assert len(infos) == 1


def test_corrupted_pdf_raises(pdfs):
    with pytest.raises(pdfp.PDFError):
        pdfp.analyze_pdf(pdfs.corrupt)


def test_non_pdf_bytes_raise():
    with pytest.raises(pdfp.PDFError):
        pdfp.analyze_pdf(b"plain text, definitely not a pdf")


def test_page_limit_enforced(pdfs):
    with pytest.raises(pdfp.PDFError, match="limit"):
        pdfp.analyze_pdf(pdfs.blank(5), max_pages=3)


def test_encrypted_pdf_rejected(pdfs):
    import fitz
    doc = fitz.open(stream=pdfs.text(["secret page with a good amount of text inside it"]), filetype="pdf")
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(pdfp.PDFError, match="password"):
        pdfp.analyze_pdf(data)


def test_ingest_uses_direct_text_without_ocr(pdfs):
    fake = FakeOCR()
    res = process_pdf_pages(pdfs.text(["Direct text page that has more than twenty five characters."]),
                            engine_override=fake)
    assert res[0].method == "direct_text" and "Direct text page" in res[0].text and fake.calls == 0


def test_ingest_ocrs_only_scanned_pages(pdfs):
    fake = FakeOCR("ocr says hello")
    data = pdfs.image([pdfs.text_image(["page one"]), pdfs.text_image(["page two"])])
    res = process_pdf_pages(data, engine_override=fake)
    assert fake.calls == 2 and all(r.text == "ocr says hello" for r in res)
    assert all(r.method == "fake" for r in res)


def test_ingest_blank_page_is_warned_not_ocrd(pdfs):
    fake = FakeOCR()
    res = process_pdf_pages(pdfs.blank(), engine_override=fake)
    assert fake.calls == 0 and res[0].text == "" and any("blank" in w.lower() for w in res[0].warnings)


def test_ingest_mixed_page_warns_about_embedded_image(pdfs):
    res = process_pdf_pages(pdfs.mixed(), engine_override=FakeOCR())
    assert res[0].kind == "mixed" and any("embedded image" in w for w in res[0].warnings)


def test_ocr_failure_on_one_page_is_recorded_not_fabricated(pdfs):
    class Flaky(FakeOCR):
        def recognize(self, image):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("engine crashed")
            return OCRResult(text="fine", engine="fake")

    res = process_pdf_pages(pdfs.image([pdfs.text_image(["a"])] * 3), engine_override=Flaky())
    assert res[0].text == "fine" and res[2].text == "fine"
    assert res[1].text == "" and "engine crashed" in res[1].error


def test_rotation_helpers_roundtrip():
    import numpy as np
    img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert pdfp.rotate_image(pdfp.rotate_image(img, 90), 270).shape == img.shape
    assert (pdfp.rotate_image(pdfp.rotate_image(img, 180), 180) == img).all()
