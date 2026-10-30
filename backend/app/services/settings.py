"""Runtime settings stored in the database, with documented defaults."""
from ..extensions import db
from ..models import Setting

DEFAULTS = {
    "ocr_engine": "tesseract",  # tesseract | qwen2vl | remote_vlm
    "ocr_model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "ocr_dpi": 200,
    "embedder": "auto",  # auto | sentence-transformers | lexical
    "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "judge": "heuristic",  # heuristic | external_llm
    "max_pages": 60,
    "max_grading_attempts": 2,
    "retention_days": 0,  # 0 = keep forever
    "mark_tolerance": 0.5,
    "min_metric_samples": 5,
    # Credit given to each concept status. These are the explicit, documented rubric
    # credits used by the grading engine; evaluators can change them here.
    "credit_correct": 1.0,
    "credit_partial": 0.5,
    "credit_mentioned": 0.25,
    "credit_incorrect": 0.0,
    "credit_missing": 0.0,
}

SETTING_CHOICES = {
    "ocr_engine": ["tesseract", "qwen2vl", "remote_vlm"],
    "embedder": ["auto", "sentence-transformers", "lexical"],
    "judge": ["heuristic", "external_llm"],
}


def get_all() -> dict:
    values = dict(DEFAULTS)
    for s in Setting.query.all():
        if s.key in DEFAULTS:
            values[s.key] = s.value
    return values


def get(key: str):
    row = db.session.get(Setting, key)
    return row.value if row is not None else DEFAULTS[key]


def validate_and_set(updates: dict) -> dict:
    errors = {}
    for key, value in updates.items():
        if key not in DEFAULTS:
            errors[key] = "unknown setting"
            continue
        default = DEFAULTS[key]
        if key in SETTING_CHOICES and value not in SETTING_CHOICES[key]:
            errors[key] = f"must be one of {SETTING_CHOICES[key]}"
            continue
        if isinstance(default, bool) or not isinstance(default, (int, float, str)):
            continue
        try:
            if isinstance(default, int) and not isinstance(default, bool):
                value = int(value)
                if value < 0:
                    raise ValueError
            elif isinstance(default, float):
                value = float(value)
                if value < 0:
                    raise ValueError
            else:
                value = str(value)
        except (TypeError, ValueError):
            errors[key] = f"invalid value for {key}"
            continue
        if key.startswith("credit_") and not (0.0 <= value <= 1.0):
            errors[key] = "credit must be between 0 and 1"
            continue
        if key == "ocr_dpi" and not (72 <= value <= 400):
            errors[key] = "ocr_dpi must be between 72 and 400"
            continue
        row = db.session.get(Setting, key)
        if row is None:
            db.session.add(Setting(key=key, value=value))
        else:
            row.value = value
    if errors:
        db.session.rollback()
        raise ValueError(errors)
    db.session.commit()
    return get_all()
