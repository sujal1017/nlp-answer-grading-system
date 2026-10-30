"""Application configuration. All values can be overridden by environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'app.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    # Run long jobs (OCR / grading) in a background thread. Tests set this to False.
    ASYNC_JOBS = _bool("ASYNC_JOBS", True)
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
    # Optional external LLM judge (OpenAI-compatible chat completions endpoint).
    # Disabled unless EXTERNAL_LLM_URL is set. Credentials are never stored in the DB.
    EXTERNAL_LLM_URL = os.environ.get("EXTERNAL_LLM_URL", "")
    EXTERNAL_LLM_KEY = os.environ.get("EXTERNAL_LLM_KEY", "")
    EXTERNAL_LLM_MODEL = os.environ.get("EXTERNAL_LLM_MODEL", "")


class TestConfig(Config):
    TESTING = True
    ASYNC_JOBS = False
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    UPLOAD_DIR = ""  # set per-test
