"""Text embedders for semantic similarity.

SentenceTransformerEmbedder is the intended production embedder (SBERT). It needs the
`sentence-transformers` package and its model weights (downloaded from Hugging Face on
first use). When unavailable, the engine falls back to LexicalEmbedder - a hashed
word/character n-gram vector model. The fallback is NOT semantic: it captures
vocabulary overlap, not meaning, so the engine reports which embedder was used and
the calling code treats its similarities more conservatively.

Similarity is only ever one signal; it never decides marks on its own.
"""
import threading
import time

import numpy as np

from .textutil import content_stems


class EmbedderUnavailable(RuntimeError):
    pass


class BaseEmbedder:
    name = "base"
    semantic = False
    hi = 0.5   # similarity at/above which a sentence is considered to express the concept
    lo = 0.3   # similarity at/above which it partially expresses it

    def encode(self, texts: list[str]) -> np.ndarray:  # L2-normalised rows
        raise NotImplementedError

    def similarity(self, a: str, b: str) -> float:
        m = self.encode([a, b])
        return float(np.clip(m[0] @ m[1], -1.0, 1.0))

    def pairwise(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        m = self.encode([query] + candidates)
        return [float(np.clip(m[0] @ row, -1.0, 1.0)) for row in m[1:]]


class LexicalEmbedder(BaseEmbedder):
    name = "lexical-fallback"
    semantic = False
    hi = 0.40
    lo = 0.20

    def __init__(self, n_features: int = 2 ** 14):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm="l2",
            analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        docs = [" ".join(content_stems(t)) or "_empty_" for t in texts]
        return self._vec.transform(docs).toarray()


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"
    semantic = True
    hi = 0.55
    lo = 0.35

    def __init__(self, model_id: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbedderUnavailable("the 'sentence-transformers' package is not installed") from exc
        try:
            self._model = SentenceTransformer(model_id)
        except Exception as exc:
            raise EmbedderUnavailable(f"could not load model '{model_id}': {exc}") from exc
        self.name = f"sentence-transformers:{model_id}"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


_cache: dict = {}
_failures: dict = {}
_lock = threading.Lock()
RETRY_AFTER_SECONDS = 300


def get_embedder(preference: str = "auto", model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> BaseEmbedder:
    """Cached embedder. 'auto' tries SBERT, then falls back to the lexical embedder."""
    with _lock:
        if preference in ("auto", "sentence-transformers"):
            key = ("st", model_id)
            if key in _cache:
                return _cache[key]
            failed_at = _failures.get(key)
            if failed_at is None or time.time() - failed_at > RETRY_AFTER_SECONDS:
                try:
                    emb = SentenceTransformerEmbedder(model_id)
                    _cache[key] = emb
                    return emb
                except EmbedderUnavailable as exc:
                    _failures[key] = time.time()
                    if preference == "sentence-transformers":
                        raise
                    _failures[(key, "msg")] = str(exc)
            elif preference == "sentence-transformers":
                raise EmbedderUnavailable(str(_failures.get((key, "msg"), "model unavailable")))
        if ("lex",) not in _cache:
            _cache[("lex",)] = LexicalEmbedder()
        return _cache[("lex",)]
