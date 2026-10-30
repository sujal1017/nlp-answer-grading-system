"""Validation of judge output before anything is saved."""
from .schema import STATUS_NEEDS_EVIDENCE, AnswerSpec, QuestionSpec
from .textutil import evidence_in_text

VALID_STATUSES = {"correct", "partial", "mentioned", "incorrect", "missing", "not_applicable"}


class ValidationFailure(ValueError):
    def __init__(self, errors: list):
        super().__init__("; ".join(errors))
        self.errors = errors


def validate_judgement(raw, question: QuestionSpec, answer: AnswerSpec) -> list[dict]:
    """Return the cleaned concept assessments or raise ValidationFailure listing every problem."""
    errors: list[str] = []
    if not isinstance(raw, dict) or not isinstance(raw.get("concepts"), list):
        raise ValidationFailure(["judge output must be an object with a 'concepts' list"])

    expected = {(crit.id, c.id) for crit in question.criteria for c in crit.concepts}
    crit_ids = {crit.id for crit in question.criteria}
    seen, cleaned = set(), []
    for i, item in enumerate(raw["concepts"]):
        if not isinstance(item, dict):
            errors.append(f"concepts[{i}] is not an object")
            continue
        cid, conc, status = item.get("criterion_id"), item.get("concept_id"), item.get("status")
        if cid not in crit_ids:
            errors.append(f"concepts[{i}]: unknown criterion_id {cid!r}")
            continue
        if (cid, conc) not in expected:
            errors.append(f"concepts[{i}]: unknown concept_id {conc!r} for criterion {cid}")
            continue
        if (cid, conc) in seen:
            errors.append(f"concept {conc!r} assessed more than once")
            continue
        seen.add((cid, conc))
        if status not in VALID_STATUSES:
            errors.append(f"concept {conc!r}: invalid status {status!r}")
            continue
        evidence = item.get("evidence")
        if status in STATUS_NEEDS_EVIDENCE:
            if not isinstance(evidence, str) or not evidence.strip():
                errors.append(f"concept {conc!r}: status '{status}' requires evidence from the answer")
                continue
            if not evidence_in_text(evidence, answer.text):
                errors.append(f"concept {conc!r}: evidence is not a quote from the student's answer")
                continue
        comment = item.get("comment")
        if not isinstance(comment, str) or not comment.strip():
            errors.append(f"concept {conc!r}: an explanation (comment) is required")
            continue
        cleaned.append({"criterion_id": cid, "concept_id": conc, "status": status,
                        "evidence": evidence.strip() if isinstance(evidence, str) and evidence.strip() else None,
                        "comment": comment.strip()})
    for cid, conc in sorted(expected - seen, key=str):
        errors.append(f"concept {conc!r} (criterion {cid}) was not assessed")
    for key in ("incorrect_statements", "flags"):
        if key in raw and not (isinstance(raw[key], list) and all(isinstance(x, str) for x in raw[key])):
            errors.append(f"'{key}' must be a list of strings")
    if errors:
        raise ValidationFailure(errors)
    return cleaned
