"""Reference Textbook / Notes PDF Ingestion and RAG Passage Retrieval Engine.
Runs 100% on CPU with zero GPU VRAM overhead using singleton all-MiniLM-L6-v2.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pymupdf as fitz

from database import DATA_DIR, update_exam_textbook
from scorer import get_embedder

TEXTBOOK_DIR = DATA_DIR / "textbooks"
TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)


def get_index_path(exam_id: str) -> Path:
    """Returns Path to {exam_id}_index.npz in TEXTBOOK_DIR."""
    eid = exam_id.strip().upper()
    return TEXTBOOK_DIR / f"{eid}_index.npz"


def index_textbook_pdf(exam_id: str, pdf_bytes: bytes, filename: str = "") -> Dict:
    """Extract clean text page by page from the uploaded reference textbook PDF,
    chunk into overlapping passages (~220 words per chunk with 40-word overlap),
    encode with all-MiniLM-L6-v2 on CPU, and save to backend/data/textbooks/{exam_id}_index.npz.
    """
    eid = exam_id.strip().upper()
    TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)
    index_path = TEXTBOOK_DIR / f"{eid}_index.npz"

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    chunks: List[Dict] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_text = page.get_text("text").strip()
        if not page_text:
            continue
        words = page_text.split()
        if not words:
            continue

        chunk_size = 220
        overlap = 40
        step = max(1, chunk_size - overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 10 and i > 0:
                break
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "text": chunk_str,
                "page": page_num,
            })
    doc.close()

    if not chunks:
        chunks.append({"text": "No text extracted from document.", "page": 1})

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    pages = [c["page"] for c in chunks]

    if embedder is not None:
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        embeddings = np.zeros((len(texts), 384), dtype=np.float32)

    np.savez_compressed(
        str(index_path),
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        pages=np.array(pages, dtype=int),
    )

    if filename:
        try:
            update_exam_textbook(eid, filename)
        except Exception as exc:
            print(f"[!] Warning: update_exam_textbook failed: {exc}")

    return {
        "status": "indexed",
        "chunks": len(chunks),
        "pages": total_pages,
        "filename": filename,
    }


def retrieve_textbook_context(exam_id: str, query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve top-k relevant textbook passages for a query via cosine similarity.
    Returns [] gracefully if no index exists for the exam.
    """
    if not exam_id or not query:
        return []

    eid = exam_id.strip().upper()
    index_path = TEXTBOOK_DIR / f"{eid}_index.npz"
    if not index_path.exists():
        return []

    try:
        with np.load(str(index_path), allow_pickle=True) as data:
            embeddings = np.array(data["embeddings"])
            texts = list(data["texts"])
            pages = list(data["pages"])

        if len(embeddings) == 0:
            return []

        embedder = get_embedder()
        if embedder is None:
            return []

        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(embeddings, q_emb)

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            results.append({
                "text": str(texts[idx]),
                "page": int(pages[idx]),
                "score": round(score, 4),
            })
        return results
    except Exception as exc:
        print(f"[!] Warning: Textbook retrieval failed for exam {eid}: {exc}")
        return []
