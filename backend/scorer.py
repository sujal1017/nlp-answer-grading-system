import os
import re
import threading
from typing import Dict, List, Optional, Set, Union
import numpy as np
from rapidfuzz import fuzz

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
        CROSS_ENCODER_AVAILABLE = False
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        CROSS_ENCODER_AVAILABLE = False

_embedder = None
_embedder_lock = threading.Lock()
_cross_encoder = None
_cross_encoder_lock = threading.Lock()

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "briefly", "explain", "describe", "define", "discuss", "give", "write", "list", "state",
    "important", "various", "major", "different", "essential"
}


def get_embedder():
    """Lazy-load sentence-transformers/all-MiniLM-L6-v2 on CPU as a singleton."""
    global _embedder
    if _embedder is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        with _embedder_lock:
            if _embedder is None:
                try:
                    # Prefer CPU to preserve RTX 4050 GPU VRAM for OCR
                    try:
                        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
                    except Exception:
                        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                except Exception as exc:
                    print(f"[!] Warning: Could not load SentenceTransformer embedder: {exc}")
                    _embedder = None
    return _embedder


def get_cross_encoder():
    """Lazy-load cross-encoder/ms-marco-MiniLM-L6-v2 on CPU as a singleton."""
    global _cross_encoder
    if _cross_encoder is None and CROSS_ENCODER_AVAILABLE:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    try:
                        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu", local_files_only=True)
                    except Exception:
                        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device="cpu")
                except Exception as exc:
                    print(f"[!] Warning: Could not load CrossEncoder: {exc}")
                    _cross_encoder = None
    return _cross_encoder


def _filter_echoed_headings(text: str, question_text: str, rubric_points: List[str]) -> str:
    """
    Strips echoed questions, rubric point headers, and heading prefixes from student answer
    to prevent students from scoring points simply by copying the question or rubric points.
    """
    if not text:
        return ""

    lines = text.splitlines()
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        cleaned_line = stripped
        # Check if line has a prefix before a colon/dash like "Rubric Title: answer text"
        for delim in [":", " - ", "—", "–"]:
            if delim in cleaned_line:
                prefix, rest = cleaned_line.split(delim, 1)
                prefix_clean = prefix.strip()
                if len(prefix_clean) > 3:
                    is_prefix_echo = False
                    if fuzz.token_set_ratio(prefix_clean, question_text) >= 82:
                        is_prefix_echo = True
                    else:
                        for pt in rubric_points:
                            if fuzz.token_set_ratio(prefix_clean, pt) >= 82:
                                is_prefix_echo = True
                                break
                    if is_prefix_echo:
                        cleaned_line = rest.strip()
                        break

        if not cleaned_line:
            continue

        # Inspect short standalone lines (< 110 chars)
        if len(cleaned_line) < 110:
            is_echo = False
            if fuzz.token_set_ratio(cleaned_line, question_text) >= 82:
                is_echo = True
            else:
                for pt in rubric_points:
                    if fuzz.token_set_ratio(cleaned_line, pt) >= 82:
                        is_echo = True
                        break
            if is_echo:
                continue

        filtered_lines.append(cleaned_line)

    return "\n".join(filtered_lines).strip()


def clean_ocr_text(text: str) -> str:
    """Clean OCR transcription: drop page headers, coordinates, tags, and collapse whitespace."""
    if not text:
        return ""
    # Strip residual page markers
    text = re.sub(r"(?:^|\n)\s*###?\s*Page\s*\d+\s*", "\n", text, flags=re.IGNORECASE)
    # Strip bounding boxes and coordinate pairs like (25,47),(998,950) or (25, 47)
    text = re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)\s*(?:,\s*\(\s*\d+\s*,\s*\d+\s*\))?", "", text)
    text = re.sub(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", "", text)
    # Strip grounding tags
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<ref>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<box>.*?</box>", "", text, flags=re.DOTALL)
    # Strip Markdown tables
    text = re.sub(r"(?:^[ \t]*\|[^\n]+\|[ \t]*(?:\r?\n|$))+", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?[ \t]*$", "", text, flags=re.MULTILINE)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_content_words(text: str) -> List[str]:
    """Extract content words (>=4 chars, letters only, non-stopwords)."""
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    return [w for w in words if w not in STOPWORDS]


def compute_fuzzy_keyword_recall(rubric_text: str, student_answer: str, threshold: float = 75.0) -> float:
    """Compute fuzzy keyword recall using RapidFuzz partial_ratio against student answer."""
    content_words = extract_content_words(rubric_text)
    if not content_words:
        return 0.5
    ans_lower = student_answer.lower()
    matched_count = 0
    for kw in content_words:
        score = fuzz.partial_ratio(kw, ans_lower)
        if score >= threshold:
            matched_count += 1
    return matched_count / len(content_words)


def _split_candidate_chunks(text: str) -> List[str]:
    """Split text into sentences, bullet lines, sliding 2-sentence windows, and full text."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 5]
    sentences = []
    for line in lines:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if len(p.strip()) > 5]
        sentences.extend(parts if parts else [line])

    if not sentences:
        sentences = [text]

    chunks = list(sentences)
    # Add sliding 2-sentence windows for multi-sentence rubric criteria
    if len(sentences) >= 2:
        for i in range(len(sentences) - 1):
            chunks.append(f"{sentences[i]} {sentences[i+1]}")

    chunks.append(text)
    return chunks


def evaluate_with_edueval(
    question_text: str,
    max_marks: float,
    key_points: List[Union[Dict, str]],
    student_answer: str,
    verified_examples: Optional[List[Dict]] = None,
    exam_id: Optional[str] = None,
) -> Dict:
    """
    EduEval Deterministic Hybrid Grader with Anti-Stuffing & Textbook Grounding:
      - Heading Echo Filter & Global Relevance Gate
      - 65% Semantic Cosine Similarity (all-MiniLM-L6-v2) + Textbook grounding boost
      - 35% Fuzzy Keyword Recall (rapidfuzz token set / partial ratio)
      - Cross-Encoder Entailment Verification (cross-encoder/ms-marco-MiniLM-L6-v2)
      - Zero GPU VRAM usage, instant deterministic evaluation on CPU.
    """
    clean_text = clean_ocr_text(student_answer)
    default_w = round(float(max_marks) / max(len(key_points), 1), 2)
    norm_points: List[Dict] = []
    for item in key_points:
        if isinstance(item, dict):
            norm_points.append({"point": item["point"], "weight": float(item.get("weight", default_w))})
        else:
            norm_points.append({"point": str(item), "weight": default_w})

    point_texts = [p["point"] for p in norm_points]

    # 1. Heading Echo Filter
    body_text = _filter_echoed_headings(clean_text, question_text, point_texts)

    if not body_text or len(body_text.strip()) < 10:
        return {
            "matched_points": [],
            "missing_points": point_texts,
            "ai_score": 0.0,
            "ai_feedback": "No scorable student answer written or answer only repeats question/headings.",
        }

    embedder = get_embedder()

    # 2. Global Question-Relevance Gate
    relevance_scale = 1.0
    if embedder is not None and body_text.strip():
        try:
            body_vec = embedder.encode([body_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            context_prompts = [question_text] + point_texts if question_text else point_texts
            ctx_vecs = embedder.encode(context_prompts, convert_to_numpy=True, normalize_embeddings=True)
            s_global = float(np.max(np.dot(ctx_vecs, body_vec)))
            if s_global < 0.30:
                relevance_scale = max(0.0, s_global / 0.30)
        except Exception as g_exc:
            relevance_scale = 1.0

    candidate_chunks = _split_candidate_chunks(body_text)

    # Pre-embed candidate chunks if embedder is available
    chunk_embs = None
    rubric_embs = None
    if embedder is not None and candidate_chunks:
        try:
            chunk_embs = embedder.encode(candidate_chunks, convert_to_numpy=True, normalize_embeddings=True)
            rubric_embs = embedder.encode(point_texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as exc:
            print(f"[!] Warning: Embedder encoding failed: {exc}")
            chunk_embs = None
            rubric_embs = None

    matched_points = []
    missing_points = []
    raw_awarded_score = 0.0

    for idx, item in enumerate(norm_points):
        pt_text = item["point"]
        weight = float(item["weight"])

        best_matching_student_window = body_text

        # a) Semantic score S_sem from cosine similarity
        if rubric_embs is not None and chunk_embs is not None:
            r_vec = rubric_embs[idx]
            sims = np.dot(chunk_embs, r_vec)
            best_chunk_idx = int(np.argmax(sims))
            best_matching_student_window = candidate_chunks[best_chunk_idx]
            max_sim = float(sims[best_chunk_idx])
            # Normalize cosine similarity from [0.25, 0.65] to [0.0, 1.0]
            s_sem = max(0.0, min(1.0, (max_sim - 0.25) / (0.65 - 0.25)))
        else:
            # Fallback when embeddings are unavailable
            s_sem = compute_fuzzy_keyword_recall(pt_text, body_text, threshold=70.0)

        # Textbook Reference Grounding Boost/Check
        if exam_id:
            try:
                from rag import retrieve_textbook_context
                book_chunks = retrieve_textbook_context(exam_id, f"{question_text} {pt_text}", top_k=2)
                if book_chunks and embedder is not None:
                    b_texts = " ".join([c["text"] for c in book_chunks if "text" in c])
                    if b_texts.strip():
                        b_emb = embedder.encode([b_texts], convert_to_numpy=True, normalize_embeddings=True)[0]
                        win_emb = embedder.encode([best_matching_student_window], convert_to_numpy=True, normalize_embeddings=True)[0]
                        sim_book = float(np.dot(win_emb, b_emb))
                        s_book = max(0.0, min(1.0, (sim_book - 0.25) / (0.65 - 0.25)))
                        s_sem = 0.75 * s_sem + 0.25 * s_book
            except Exception:
                pass

        # b) Fuzzy keyword score S_kw
        s_kw = compute_fuzzy_keyword_recall(pt_text, body_text, threshold=75.0)

        # c) Combined match metric M_i
        m_i = 0.65 * s_sem + 0.35 * s_kw

        # Apply global relevance scale
        m_i = m_i * relevance_scale

        # d) Cross-Encoder Entailment Verification
        is_matched = False
        if m_i >= 0.35:
            ce = get_cross_encoder()
            if ce is not None:
                try:
                    pairs = [
                        (f"{question_text}: {pt_text}", best_matching_student_window),
                        (f"{question_text}: {pt_text}", body_text),
                        (pt_text, best_matching_student_window),
                        (pt_text, body_text),
                    ]
                    logits = ce.predict(pairs)
                    max_logit = float(np.max(logits))
                    # Calibrated sigmoid with MS-MARCO margin adjustment for answer snippets
                    ce_score = 1.0 / (1.0 + np.exp(-(max_logit + 2.2)))
                except Exception:
                    ce_score = 1.0
            else:
                ce_score = 1.0

            if ce_score >= 0.35:
                is_matched = True

        if is_matched:
            matched_points.append(pt_text)
            if m_i >= 0.60:
                awarded = weight
            else:
                awarded = weight * min(1.0, m_i / 0.60)
            raw_awarded_score += awarded
        else:
            missing_points.append(pt_text)

    # Length & Elaboration Bonus: Detailed multi-paragraph answer (>=120 words) with >=75% match
    word_count = len(body_text.split())
    if word_count >= 120 and len(matched_points) >= max(1, int(0.75 * len(norm_points))):
        raw_awarded_score = max(raw_awarded_score, float(max_marks) * 0.9)

    # Few-Shot Teacher Calibration Adjustment
    if verified_examples:
        biases = []
        for ex in verified_examples:
            t_score = ex.get("teacher_score")
            a_score = ex.get("ai_score")
            if t_score is not None and a_score is not None:
                try:
                    biases.append(float(t_score) - float(a_score))
                except (ValueError, TypeError):
                    pass
        if biases:
            avg_bias = sum(biases) / len(biases)
            raw_awarded_score += 0.5 * avg_bias

    # Clamp and round to nearest 0.5
    final_score = round(min(max(raw_awarded_score, 0.0), float(max_marks)) * 2) / 2

    # Deterministic Explainability Feedback (2 clear sentences)
    matched_count = len(matched_points)
    total_count = len(norm_points)
    if matched_count == total_count:
        feedback = (
            f"Satisfied all {total_count} rubric criteria with strong conceptual coverage and clear elaboration. "
            f"Demonstrates comprehensive mastery of the topic."
        )
    elif matched_count > 0:
        first_m = matched_points[0]
        first_miss = missing_points[0] if missing_points else ""
        if first_miss:
            feedback = (
                f"Satisfied {matched_count} of {total_count} rubric criteria, clearly covering '{first_m[:50]}...'. "
                f"To improve the score, expand on '{first_miss[:50]}...'."
            )
        else:
            feedback = (
                f"Satisfied {matched_count} of {total_count} rubric criteria with good structure across major aspects. "
                f"Solid conceptual understanding shown in the handwritten script."
            )
    else:
        feedback = (
            f"The answer does not adequately address the {total_count} expected rubric criteria for this question. "
            f"Key conceptual components and required academic terminology were missing."
        )

    return {
        "matched_points": matched_points,
        "missing_points": missing_points,
        "ai_score": final_score,
        "ai_feedback": feedback,
    }
