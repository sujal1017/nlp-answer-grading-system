"""Rubric-based grading engine - independent of Flask and the database.

Flow: pre-checks -> judge assesses concepts -> validate -> backend computes marks from
the rubric -> narrative feedback built from the concrete assessments.
If anything cannot be done reliably the outcome is 'manual_required' with NO marks.
"""
from typing import Optional

from .schema import (AnswerSpec, ConceptAssessment, CriterionAssessment, EvaluationResult, Outcome,
                     QuestionSpec)
from .judges import JudgeError
from .validation import ValidationFailure, validate_judgement

DEFAULT_CREDITS = {"correct": 1.0, "partial": 0.5, "mentioned": 0.25, "incorrect": 0.0, "missing": 0.0}
STEP = 0.25  # marks are awarded in quarter-mark steps


def _round_step(x: float) -> float:
    return round(x / STEP) * STEP


def precheck(question: QuestionSpec, answer: AnswerSpec, judge) -> Optional[str]:
    """Return a reason the question cannot be graded automatically, or None."""
    if question.max_marks is None or question.max_marks <= 0:
        return "Maximum marks for this question are not confirmed."
    if not question.reference_answer.strip():
        return "No reference answer has been provided."
    if not question.reference_approved:
        return "The reference answer is an unapproved AI draft; approve or replace it before grading."
    if not question.criteria:
        return "No marking rubric (criteria) has been defined."
    total = round(sum(c.max_marks for c in question.criteria), 4)
    if abs(total - question.max_marks) > 1e-6:
        return f"Rubric criteria add up to {total:g} marks but the question is worth {question.max_marks:g}."
    empty = [c.name for c in question.criteria if not c.concepts]
    if empty:
        return f"Criteria without any expected concepts cannot be assessed: {', '.join(empty)}."
    if question.qtype not in judge.supported_qtypes:
        return (f"'{question.qtype}' questions need method/calculation checking that the "
                f"'{judge.name}' judge cannot do reliably. Grade manually or configure an external judge.")
    if answer.status == "unreadable":
        return "The answer could not be read (OCR failed). Transcribe it manually or re-run OCR."
    if answer.status == "ambiguous" and not answer.mapping_confirmed:
        return "The mapping of this answer to the question is unconfirmed."
    return None


def compute_marks(question: QuestionSpec, assessed: list[dict], credits: dict) -> tuple[list[CriterionAssessment], float]:
    by_crit: dict[int, list[dict]] = {}
    for a in assessed:
        by_crit.setdefault(a["criterion_id"], []).append(a)
    concept_text = {c.id: c.text for crit in question.criteria for c in crit.concepts}
    out, total = [], 0.0
    for crit in question.criteria:
        items = by_crit.get(crit.id, [])
        applicable = [a for a in items if a["status"] != "not_applicable"]
        if applicable:
            frac = sum(credits[a["status"]] for a in applicable) / len(applicable)
            awarded = min(max(_round_step(crit.max_marks * frac), 0.0), crit.max_marks)
        else:
            awarded = 0.0
        counts = {}
        for a in applicable:
            counts[a["status"]] = counts.get(a["status"], 0) + 1
        summary = ", ".join(f"{n} {s}" for s, n in counts.items()) or "no applicable concepts"
        out.append(CriterionAssessment(
            criterion_id=crit.id, name=crit.name, awarded=awarded, max_marks=crit.max_marks,
            concepts=[ConceptAssessment(concept_text=concept_text.get(a["concept_id"], ""), **a) for a in items],
            rationale=f"{awarded:g}/{crit.max_marks:g} marks: {summary}.",
        ))
        total += awarded
    total = min(total, question.max_marks)
    return out, total


def build_feedback(question: QuestionSpec, criteria: list[CriterionAssessment], extra_incorrect: list[str]):
    awarded, withheld, improve, outline, incorrect = [], [], [], [], list(extra_incorrect)
    for crit in criteria:
        for a in crit.concepts:
            label = a.concept_text or a.concept_id
            outline.append(f"{crit.name}: {label}")
            ev = f' (you wrote: "{a.evidence}")' if a.evidence else ""
            if a.status == "correct":
                awarded.append(f"{crit.name}: '{label}' is explained{ev}.")
            elif a.status == "partial":
                withheld.append(f"{crit.name}: '{label}' is only partly explained{ev}.")
                improve.append(f"Complete your explanation of '{label}' - what you wrote covers only part of it.")
            elif a.status == "mentioned":
                withheld.append(f"{crit.name}: '{label}' is mentioned but not explained{ev}.")
                improve.append(f"Don't just name '{label}' - explain what it means and how it works.")
            elif a.status == "incorrect":
                withheld.append(f"{crit.name}: your statement about '{label}' appears incorrect{ev}.")
                improve.append(f"Re-check your statement on '{label}'; it conflicts with the expected answer.")
            elif a.status == "missing":
                withheld.append(f"{crit.name}: '{label}' is missing from the answer.")
                improve.append(f"Add '{label}' to your answer.")
    return awarded, withheld, improve, outline, incorrect


class GradingEngine:
    def __init__(self, judge, credits: Optional[dict] = None, max_attempts: int = 2, embedder_name: str = ""):
        self.judge = judge
        self.credits = {**DEFAULT_CREDITS, **(credits or {})}
        self.max_attempts = max(1, max_attempts)
        self.embedder_name = embedder_name

    def evaluate(self, question: QuestionSpec, answer: AnswerSpec) -> Outcome:
        base = dict(judge_name=self.judge.name, embedder_name=self.embedder_name)
        reason = precheck(question, answer, self.judge)
        if reason:
            return Outcome(status="manual_required", error=reason, **base)

        flags: list[str] = []
        if answer.status == "unanswered" or not answer.text.strip():
            result = self._empty_result(question)
            result.flags = ["unanswered: no answer text was found for this question; confirm it was not attempted"]
            return Outcome(status="needs_review", result=result, flags=result.flags, attempts=0, **base)

        errors: list[str] = []
        raw = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.judge.assess(question, answer)
                assessed = validate_judgement(raw, question, answer)
                break
            except ValidationFailure as exc:
                errors = exc.errors
                assessed = None
            except JudgeError as exc:
                return Outcome(status="manual_required", error=f"Judge unavailable: {exc}", attempts=attempt, **base)
        else:
            return Outcome(status="manual_required", attempts=self.max_attempts,
                           error="AI output failed validation after retries: " + "; ".join(errors[:6]), **base)

        criteria, total = compute_marks(question, assessed, self.credits)
        a, w, imp, outline, incorrect = build_feedback(question, criteria, raw.get("incorrect_statements", []))
        flags.extend(raw.get("flags", []))
        if len(answer.text.split()) < 4:
            flags.append("very_short_answer: the answer is only a few words")
        if any(c.status == "incorrect" for cr in criteria for c in cr.concepts):
            flags.append("contains_possibly_incorrect_statements: verify before approving")
        if answer.status == "ambiguous":
            flags.append("mapping_was_ambiguous: confirmed by reviewer")
        result = EvaluationResult(
            question_id=question.id, total=total, max_total=question.max_marks, criteria=criteria,
            awarded_because=a, withheld_because=w, incorrect_statements=incorrect, improvements=imp,
            stronger_answer_outline=outline, flags=flags,
        )
        status = "needs_review" if flags else "ai_graded"
        return Outcome(status=status, result=result, flags=flags, attempts=attempt, **base)

    def _empty_result(self, question: QuestionSpec) -> EvaluationResult:
        criteria, outline, withheld = [], [], []
        for crit in question.criteria:
            concepts = [ConceptAssessment(concept_id=c.id, concept_text=c.text, criterion_id=crit.id,
                                          status="missing", evidence=None, comment="No answer was found.")
                        for c in crit.concepts]
            for c in crit.concepts:
                outline.append(f"{crit.name}: {c.text}")
                withheld.append(f"{crit.name}: '{c.text}' is missing (no answer found).")
            criteria.append(CriterionAssessment(criterion_id=crit.id, name=crit.name, awarded=0.0,
                                                max_marks=crit.max_marks, concepts=concepts,
                                                rationale=f"0/{crit.max_marks:g} marks: no answer."))
        return EvaluationResult(question_id=question.id, total=0.0, max_total=question.max_marks,
                                criteria=criteria, withheld_because=withheld,
                                improvements=["Attempt this question; see the outline of expected content."],
                                stronger_answer_outline=outline)
