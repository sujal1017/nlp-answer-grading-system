import threading
from typing import Dict, List, Tuple, Union

MODEL_ID = "textattack/bert-base-uncased-MNLI"
_model = None
_tokenizer = None
_torch = None
_load_lock = threading.Lock()


def _load_model():
    global _model, _tokenizer, _torch
    if _model is None:
        with _load_lock:
            if _model is None:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                _model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
                _model.to("cpu")
                _model.eval()
                _torch = torch
    return _model, _tokenizer, _torch


def _entailment_index(model) -> int:
    id2label = model.config.id2label
    for index, label in id2label.items():
        normalized = str(label).lower().replace("_", " ").replace("-", " ")
        if "entail" in normalized:
            return int(index)
    if (
        getattr(model.config, "finetuning_task", None) == "mnli"
        and all(str(label) == f"LABEL_{index}" for index, label in id2label.items())
    ):
        return 1
    raise RuntimeError(f"{MODEL_ID} does not expose an entailment label.")


def _score_entailment_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
    model, tokenizer, torch = _load_model()
    entailment_index = _entailment_index(model)
    scores = []

    for start in range(0, len(pairs), 16):
        batch = pairs[start : start + 16]
        inputs = tokenizer(
            [premise for premise, _ in batch],
            [hypothesis for _, hypothesis in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, entailment_index]
        scores.extend(probabilities.tolist())

    return [float(score) for score in scores]


def evaluate_with_bert(
    question_text: str,
    max_marks: float,
    key_points: List[Union[Dict, str]],
    student_answer: str,
) -> Dict:
    """Score each rubric point by its maximum BERT-MNLI entailment probability."""
    from scorer import _split_candidate_chunks, clean_ocr_text

    clean_text = clean_ocr_text(student_answer)
    default_weight = float(max_marks) / max(len(key_points), 1)
    rubric_points = []
    for item in key_points:
        if isinstance(item, dict):
            rubric_points.append({
                "point": str(item["point"]),
                "weight": float(item.get("weight", default_weight)),
            })
        else:
            rubric_points.append({"point": str(item), "weight": default_weight})

    if not clean_text or len(clean_text) < 10:
        return {
            "matched_points": [],
            "missing_points": [item["point"] for item in rubric_points],
            "ai_score": 0.0,
            "ai_feedback": "No student answer written or transcription is empty.",
        }

    candidate_chunks = _split_candidate_chunks(clean_text)
    pairs = [
        (f"Question: {question_text}\nStudent answer: {chunk}", item["point"])
        for item in rubric_points
        for chunk in candidate_chunks
    ]
    pair_scores = _score_entailment_pairs(pairs) if pairs else []

    matched_points = []
    missing_points = []
    raw_score = 0.0
    offset = 0
    for item in rubric_points:
        scores = pair_scores[offset : offset + len(candidate_chunks)]
        offset += len(candidate_chunks)
        entailment = max(scores, default=0.0)
        raw_score += item["weight"] * entailment
        if entailment >= 0.5:
            matched_points.append(item["point"])
        else:
            missing_points.append(item["point"])

    final_score = round(min(max(raw_score, 0.0), float(max_marks)) * 2) / 2
    if matched_points:
        feedback = (
            f"BERT (MNLI) found entailment for {len(matched_points)} of "
            f"{len(rubric_points)} rubric criteria."
        )
    else:
        feedback = (
            f"BERT (MNLI) found no rubric criteria clearly entailed by the answer "
            f"({len(rubric_points)} criteria checked)."
        )

    return {
        "matched_points": matched_points,
        "missing_points": missing_points,
        "ai_score": final_score,
        "ai_feedback": feedback,
    }
