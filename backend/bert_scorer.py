import re
import threading
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from rapidfuzz import fuzz

MODEL_ID = "sentence-transformers/bert-base-nli-mean-tokens"
_model = None
_tokenizer = None
_torch = None
_device = "cpu"
_model_lock = threading.RLock()

STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before",
    "being", "between", "both", "but", "can", "could", "each", "for", "from",
    "have", "into", "its", "more", "most", "only", "other", "should", "that",
    "the", "their", "there", "these", "this", "those", "through", "under", "using",
    "was", "were", "what", "when", "where", "which", "while", "with", "would",
}


def _load_model():
    global _model, _tokenizer, _torch, _device
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                model = AutoModel.from_pretrained(MODEL_ID)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    try:
                        model.to(device)
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        torch.cuda.empty_cache()
                        device = "cpu"
                        model.to(device)
                else:
                    model.to(device)

                model.eval()
                _model = model
                _tokenizer = tokenizer
                _torch = torch
                _device = device
    return _model, _tokenizer, _torch


def _encode_texts(texts: List[str]) -> np.ndarray:
    global _device
    model, tokenizer, torch = _load_model()
    encoded_batches = []

    for start in range(0, len(texts), 8):
        batch = texts[start : start + 8]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        try:
            with _model_lock:
                inputs = {name: tensor.to(_device) for name, tensor in inputs.items()}
                with torch.inference_mode():
                    hidden = model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        except RuntimeError as exc:
            if _device != "cuda" or "out of memory" not in str(exc).lower():
                raise
            with _model_lock:
                model.to("cpu")
                _device = "cpu"
                torch.cuda.empty_cache()
                cpu_inputs = {
                    name: tensor.cpu() for name, tensor in inputs.items()
                }
                with torch.inference_mode():
                    hidden = model(**cpu_inputs).last_hidden_state
                    mask = cpu_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        encoded_batches.append(embeddings.cpu().numpy())

    return np.concatenate(encoded_batches, axis=0) if encoded_batches else np.empty((0, 768))


def _text_chunks(text: str) -> List[str]:
    chunks = []
    for line in (text or "").splitlines():
        chunks.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", line)
            if sentence.strip()
        )
    if not chunks and text.strip():
        chunks.append(text.strip())
    return chunks


def _keywords(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", (text or "").lower())
    return list(dict.fromkeys(word for word in words if word not in STOPWORDS))


def _keyword_coverage(keywords: List[str], answer: str) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    return sum(fuzz.partial_ratio(word, answer_lower) >= 85 for word in keywords) / len(keywords)


def evaluate_with_bert(
    question_text: str,
    max_marks: float,
    key_points: List[Union[Dict, str]],
    student_answer: str,
    reference_answer: str = "",
) -> Dict:
    """Grade an answer with BERT-NLI sentence embeddings plus rubric keyword coverage."""
    answer = (student_answer or "").strip()
    max_marks = max(float(max_marks), 0.0)
    if not answer or max_marks == 0:
        return {
            "matched_points": [],
            "missing_points": [item["point"] if isinstance(item, dict) else str(item) for item in key_points],
            "ai_score": 0.0,
            "similarity_score": 0.0,
            "keyword_coverage": 0.0,
            "ai_feedback": "No student answer was available to grade.",
        }

    default_weight = max_marks / max(len(key_points), 1)
    rubric = []
    for item in key_points:
        if isinstance(item, dict):
            point = str(item.get("point", "")).strip()
            weight = max(float(item.get("weight", default_weight)), 0.0)
        else:
            point = str(item).strip()
            weight = default_weight
        if point:
            rubric.append({"point": point, "weight": weight})

    if not rubric:
        fallback_point = reference_answer.strip() or question_text.strip()
        if fallback_point:
            rubric = [{"point": fallback_point, "weight": max_marks}]
    if not rubric:
        return {
            "matched_points": [],
            "missing_points": [],
            "ai_score": 0.0,
            "similarity_score": 0.0,
            "keyword_coverage": 0.0,
            "ai_feedback": "No reference answer or rubric criteria are configured.",
        }

    answer_chunks = _text_chunks(answer)
    reference_text = reference_answer.strip() or " ".join(item["point"] for item in rubric)
    reference_chunks = _text_chunks(reference_text) or [reference_text]
    answer_embeddings = _encode_texts(answer_chunks)
    reference_embeddings = _encode_texts(reference_chunks)
    criterion_embeddings = _encode_texts([item["point"] for item in rubric])

    reference_similarities = answer_embeddings @ reference_embeddings.T
    reference_similarity = float(np.clip(reference_similarities.max(axis=0).mean(), 0.0, 1.0))

    matched_points = []
    missing_points = []
    criterion_scores = []
    total_weight = sum(item["weight"] for item in rubric)
    if total_weight <= 0:
        for item in rubric:
            item["weight"] = max_marks / len(rubric)
        total_weight = max_marks

    for item, criterion_embedding in zip(rubric, criterion_embeddings):
        semantic_similarity = float(
            np.clip((answer_embeddings @ criterion_embedding).max(), 0.0, 1.0)
        )
        terms = _keywords(item["point"])
        keyword_coverage = _keyword_coverage(terms, answer)
        criterion_score = 0.75 * semantic_similarity + 0.25 * keyword_coverage
        criterion_scores.append((item["weight"] / total_weight, criterion_score))
        if criterion_score >= 0.5:
            matched_points.append(item["point"])
        else:
            missing_points.append(item["point"])

    rubric_score = sum(weight * score for weight, score in criterion_scores)
    overall_coverage = _keyword_coverage(
        _keywords(" ".join(item["point"] for item in rubric)), answer
    )
    score_fraction = 0.8 * rubric_score + 0.2 * reference_similarity
    score = round(min(max(score_fraction * max_marks, 0.0), max_marks) * 2) / 2
    score = min(score, max_marks)

    feedback = (
        f"Awarded {score:g}/{max_marks:g}. BERT semantic similarity to the reference is "
        f"{reference_similarity:.0%}, and rubric keyword coverage is {overall_coverage:.0%}. "
    )
    if missing_points:
        feedback += "Review these criteria: " + "; ".join(missing_points[:3]) + "."
    elif rubric:
        feedback += "All configured rubric criteria were addressed."

    return {
        "matched_points": matched_points,
        "missing_points": missing_points,
        "ai_score": score,
        "similarity_score": round(reference_similarity, 4),
        "keyword_coverage": round(overall_coverage, 4),
        "ai_feedback": feedback,
    }


def unload_bert_model():
    global _model, _tokenizer, _torch, _device
    with _model_lock:
        if _model is not None:
            del _model
            _model = None
            _tokenizer = None
            if _torch is not None and _device == "cuda":
                _torch.cuda.empty_cache()
        _device = "cpu"


def bert_model_loaded() -> bool:
    return _model is not None
