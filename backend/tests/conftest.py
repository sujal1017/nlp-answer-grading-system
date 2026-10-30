"""Shared fixtures. PDFs are generated on the fly so tests need no binary assets, except the
real handwriting photographs in tests/fixtures/handwriting (taken from the project's qwen-ocr branch)."""
import io
import os
import sys
from pathlib import Path

import pymupdf as fitz
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app  # noqa: E402
from app.config import TestConfig  # noqa: E402
from app.extensions import db  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def app(tmp_path):
    class Cfg(TestConfig):
        UPLOAD_DIR = str(tmp_path / "uploads")
        SQLALCHEMY_DATABASE_URI = "sqlite://"

    app = create_app(Cfg)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


# ----------------------------------------------------------------- PDF builders
def text_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        y = 60
        for line in text.splitlines():
            page.insert_text((50, y), line, fontsize=11)
            y += 16
    data = doc.tobytes()
    doc.close()
    return data


def text_image(lines: list[str], size=(1240, 1754), font_size=34, rotate=0) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=font_size)
    y = 80
    for line in lines:
        draw.text((70, y), line, fill="black", font=font)
        y += int(font_size * 1.6)
    return img.rotate(rotate, expand=True) if rotate else img


def image_pdf(images: list[Image.Image]) -> bytes:
    """A 'scanned' PDF: each page is only a picture, no text layer."""
    doc = fitz.open()
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


def mixed_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 60), "Typed header with plenty of selectable words on this page.", fontsize=11)
    buf = io.BytesIO()
    text_image(["handwritten part"], size=(800, 500)).save(buf, format="PNG")
    page.insert_image(fitz.Rect(40, 100, 555, 500), stream=buf.getvalue())
    data = doc.tobytes()
    doc.close()
    return data


def blank_pdf(n=1) -> bytes:
    doc = fitz.open()
    for _ in range(n):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


QUESTION_PAPER = """SDM College of Engineering & Technology
Internal Assessment - I
Note: Answer any one full question from Q.No.1 & Q.No.2.; Q.No.3 is COMPULSORY.
Q. No. 1
a) What is a distributed system? Discuss the goals of a distributed system. [5M, CO:1]
b) Explain the different types of transparency in distributed systems. [5M, CO:1]
Q. No. 2
a) Explain the Bully election algorithm. [5M, CO:2]
b) Explain the Ring election algorithm. [5M, CO:2]
Q. No. 3
a) Explain the Network Time Protocol. [5M, CO:2]"""


@pytest.fixture()
def pdfs():
    class P:
        text = staticmethod(text_pdf)
        image = staticmethod(image_pdf)
        mixed = staticmethod(mixed_pdf)
        blank = staticmethod(blank_pdf)
        text_image = staticmethod(text_image)
        question_paper = QUESTION_PAPER
        corrupt = b"%PDF-1.4\nthis is not really a pdf \x00\x01\x02"
    return P


def upload(client, url, data: bytes, filename="file.pdf", **form):
    return client.post(url, data={"file": (io.BytesIO(data), filename), **form},
                       content_type="multipart/form-data")


@pytest.fixture(autouse=True)
def _restore_ocr_engine_cache():
    """Mock OCR engines registered by a test must never leak into later tests (real-OCR tests)."""
    from app.services import ocr as _ocr
    with _ocr._engines_lock:
        saved = dict(_ocr._engines)
    yield
    with _ocr._engines_lock:
        _ocr._engines.clear()
        _ocr._engines.update(saved)
