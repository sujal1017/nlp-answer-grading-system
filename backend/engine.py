import re
from typing import Dict, List, Optional, Union

from handwriting_ocr import HandwritingOCREngine, MODEL_ID as TROCR_MODEL_ID

BLOOM_TEMPLATES = {
    "define": [
        "Accurate definition and core meaning of {topic}",
        "Essential characteristics / distinguishing features of {topic}",
        "Relevant classification, types, or related terms",
        "Example or academic significance of {topic}",
    ],
    "explain": [
        "Clear explanation of the meaning of {topic}",
        "Underlying process, reasons, or working of {topic}",
        "Key factors, variables, or components involved",
        "Implications, outcome, or academic relevance",
    ],
    "describe": [
        "Systematic description of {topic}",
        "Ordered steps, stages, or procedure",
        "Purpose of each major step",
        "Expected outcome or use in research practice",
    ],
    "discuss": [
        "Concept and scope of {topic}",
        "Arguments, viewpoints, or theoretical basis",
        "Applications or illustrations in research",
        "Limitations, issues, or concluding remark",
    ],
    "compare": [
        "Identification of items being compared in {topic}",
        "Similarities with supporting points",
        "Differences with supporting points",
        "Justified conclusion or preference",
    ],
    "differentiate": [
        "Clear identification of the two sides in {topic}",
        "Contrasting attributes presented in parallel",
        "Examples that make the distinction concrete",
        "Summary of why the distinction matters",
    ],
    "list": [
        "Coverage of the required items in {topic}",
        "Correct naming / terminology of each item",
        "Brief elaboration of the listed points",
        "Organisation and completeness of the list",
    ],
    "evaluate": [
        "Criteria used to evaluate {topic}",
        "Balanced strengths and weaknesses",
        "Evidence or examples supporting the judgement",
        "A reasoned concluding assessment",
    ],
}

VERB_ORDER = [
    "differentiate", "compare", "evaluate", "discuss", "describe",
    "explain", "define", "list", "state", "write",
]


def calibrate_weights(n: int, max_marks: float) -> List[float]:
    n = max(n, 1)
    base = round(float(max_marks) / n, 2)
    weights = [base] * n
    drift = round(float(max_marks) - sum(weights), 2)
    weights[0] = round(weights[0] + drift, 2)
    return weights


def generate_atomic_rubric(question_text: str, max_marks: float, num_points: int = 4) -> List[Dict]:
    """Create editable rubric criteria and weights from the teacher's question."""
    question = (question_text or "").strip()
    verb = "explain"
    for candidate in VERB_ORDER:
        if re.search(rf"\b{candidate}\b", question, re.I):
            verb = "list" if candidate in {"list", "state", "write"} else candidate
            if verb not in BLOOM_TEMPLATES:
                verb = "explain"
            break

    topic = re.sub(
        r"^\s*(briefly\s+)?(define|explain|describe|discuss|compare|differentiate|list|state|write|evaluate)\b[\s:,-]*",
        "",
        question,
        flags=re.I,
    ).strip(" .:?")
    if not topic:
        topic = question[:80] or "the asked concept"

    templates = BLOOM_TEMPLATES.get(verb, BLOOM_TEMPLATES["explain"])[:num_points]
    points = [template.format(topic=topic) for template in templates]
    weights = calibrate_weights(len(points), max_marks)
    return [{"point": point, "weight": weight} for point, weight in zip(points, weights)]


class LocalGradingEngine:
    """Lazy TrOCR and BERT services; OCR is released before BERT grading begins."""

    def __init__(self):
        self.ocr = HandwritingOCREngine()

    def transcribe_images(self, image_paths, progress_cb=None, auto_unload=True):
        from bert_scorer import unload_bert_model

        unload_bert_model()
        return self.ocr.transcribe_images(
            image_paths,
            progress_cb=progress_cb,
            auto_unload=auto_unload,
        )

    def evaluate_answer(
        self,
        question_text: str,
        max_marks: float,
        key_points: List[Union[Dict, str]],
        student_answer: str,
        reference_answer: str = "",
    ) -> Dict:
        from bert_scorer import evaluate_with_bert

        self.ocr.unload_model()
        return evaluate_with_bert(
            question_text=question_text,
            max_marks=max_marks,
            key_points=key_points,
            student_answer=student_answer,
            reference_answer=reference_answer,
        )

    def unload_model(self):
        from bert_scorer import unload_bert_model

        self.ocr.unload_model()
        unload_bert_model()

    def telemetry(self) -> Dict:
        import torch
        from bert_scorer import MODEL_ID as BERT_MODEL_ID, bert_model_loaded

        cuda_available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        return {
            "device": self.ocr.device,
            "device_name": device_name,
            "cuda_available": cuda_available,
            "ocr_model": TROCR_MODEL_ID,
            "ocr_loaded": self.ocr.model is not None,
            "ocr_status": "Loaded" if self.ocr.model is not None else "Ready on first use",
            "grader_model": BERT_MODEL_ID,
            "grader_loaded": bert_model_loaded(),
            "grader_status": "Loaded" if bert_model_loaded() else "Ready on first use",
        }
