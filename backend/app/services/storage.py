"""Controlled file storage: server-generated names, path-traversal protection, validation."""
import hashlib
import os
import re
import uuid
from pathlib import Path

from flask import current_app


class UploadError(ValueError):
    """Raised for invalid uploads; message is safe to show to users."""


def safe_display_name(filename: str) -> str:
    """Sanitised version of the user's filename, for display only (never used as a path)."""
    name = os.path.basename(filename or "").replace("\x00", "")
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return (name or "upload.pdf")[:200]


def upload_root() -> Path:
    root = Path(current_app.config["UPLOAD_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def save_pdf_bytes(data: bytes) -> tuple[str, str]:
    """Persist PDF bytes under a random name. Returns (stored_name, sha256)."""
    stored = f"{uuid.uuid4().hex}.pdf"
    (upload_root() / stored).write_bytes(data)
    return stored, hashlib.sha256(data).hexdigest()


def resolve_stored(stored_name: str) -> Path:
    """Resolve a stored name to a path, refusing anything outside the upload root."""
    if not re.fullmatch(r"[0-9a-f]{32}\.pdf", stored_name or ""):
        raise UploadError("invalid stored file name")
    root = upload_root()
    path = (root / stored_name).resolve()
    if root not in path.parents:
        raise UploadError("path escapes upload directory")
    return path


def validate_pdf_upload(file_storage) -> bytes:
    """Validate an uploaded file is a non-empty, size-limited PDF (by content, not extension)."""
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise UploadError("no file uploaded")
    name = file_storage.filename.lower()
    if not name.endswith(".pdf"):
        raise UploadError("only PDF files are supported")
    data = file_storage.read()
    limit = current_app.config["MAX_UPLOAD_MB"] * 1024 * 1024
    if not data:
        raise UploadError("the uploaded file is empty")
    if len(data) > limit:
        raise UploadError(f"file exceeds the {current_app.config['MAX_UPLOAD_MB']} MB limit")
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        raise UploadError("file content is not a valid PDF")
    return data
