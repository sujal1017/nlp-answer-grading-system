"""Orchestration between the database and the pure processing/grading modules."""
import re
from datetime import datetime, timezone
from typing import Optional

from flask import current_app

from ..extensions import db
from ..grading.embeddings import get_embedder
from ..grading.engine import GradingEngine
from ..grading.judges import ExternalLLMJudge, HeuristicJudge, JudgeError
from ..grading.schema import AnswerSpec, ConceptSpec, CriterionSpec, QuestionSpec
from ..models import Answer, Criterion, Document, Evaluation, PageText, Question, Submission
from . import settings as settings_svc
from .answer_mapping import map_answers
from .ingest import process_pdf_pages
from .question_extraction import detect_qtype, parse_question_paper
from .storage import resolve_stored

ILLEGIBLE = "[illegible]"


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ------------------------------------------------------------------ question paper
def extract_question_paper(exam, document: Document, progress=None) -> dict:
    cfg = settings_svc.get_all()
    data = resolve_stored(document.stored_name).read_bytes()
    pages = process_pdf_pages(data, cfg["ocr_engine"], cfg["ocr_model_id"], cfg["ocr_dpi"],
                              cfg["max_pages"], progress)
    errors = [f"Page {p.page_number}: {p.error}" for p in pages if p.error]
    parsed = parse_question_paper([(p.page_number, p.text) for p in pages if p.text])

    # Replace earlier auto-extracted questions that nothing depends on; never touch manual ones.
    for q in list(exam.questions):
        if q.source == "extracted" and not Answer.query.filter_by(question_id=q.id).first():
            db.session.delete(q)
    db.session.flush()
    existing = {q.label for q in exam.questions}
    position = max([q.position for q in exam.questions], default=-1) + 1
    created = []
    for item in parsed["questions"]:
        label = item["label"]
        while label in existing:
            label += "_x"
            item["review_reasons"].append("Label collided with an existing question; renamed.")
            item["needs_review"] = True
        existing.add(label)
        q = Question(
            exam_id=exam.id, position=position, label=label, text=item["text"], context=item["context"],
            max_marks=item["max_marks"], marks_confirmed=False, qtype=item["qtype"],
            source_page=item["source_page"], source="extracted", needs_review=item["needs_review"],
            review_reasons=item["review_reasons"],
        )
        position += 1
        db.session.add(q)
        created.append(q)
    if parsed["instructions"]:
        exam.instructions = parsed["instructions"]
    document.page_analysis = [
        {"page": p.page_number, "kind": p.kind, "method": p.method, "chars": len(p.text),
         "warnings": p.warnings, "error": p.error} for p in pages
    ]
    db.session.commit()
    warnings = list(errors)
    if not created:
        warnings.append("No questions could be extracted automatically; add them manually.")
    return {"created": len(created), "warnings": warnings}


# ------------------------------------------------------------------ answer sheets
def run_answer_sheet_ocr(submission_id: int, progress=None, engine_override=None):
    sub = db.session.get(Submission, submission_id)
    cfg = settings_svc.get_all()
    sub.status = "ocr_running"
    db.session.commit()
    data = resolve_stored(sub.document.stored_name).read_bytes()
    try:
        pages = process_pdf_pages(data, cfg["ocr_engine"], cfg["ocr_model_id"], cfg["ocr_dpi"],
                                  cfg["max_pages"], progress, engine_override)
    except Exception:
        sub.status = "ocr_failed"
        db.session.commit()
        raise
    PageText.query.filter_by(submission_id=sub.id).delete()
    for p in pages:
        db.session.add(PageText(
            submission_id=sub.id, page_number=p.page_number, page_kind=p.kind, method=p.method,
            engine_info=p.engine_info, text_original=p.text, confidence=p.confidence,
            warnings=p.warnings, error=p.error,
        ))
    sub.status = "ocr_done" if any(not p.error for p in pages) else "ocr_failed"
    db.session.commit()
    if sub.status == "ocr_done":
        map_submission(sub)


def map_submission(sub: Submission) -> None:
    """(Re)build Answer rows from the current page texts, preserving human corrections."""
    questions = Question.query.filter_by(exam_id=sub.exam_id).order_by(Question.position).all()
    pages = [(p.page_number, p.effective_text) for p in sub.pages if p.effective_text]
    page_errors = any(p.error for p in sub.pages)
    result = map_answers(pages, [q.label for q in questions])
    by_page = {p.page_number: p for p in sub.pages}
    notes = list(result.notes)
    if result.unmapped_text:
        notes.append("Some text could not be matched to any question (see 'unmapped text').")
    sub.unmapped_text = result.unmapped_text
    existing = {a.question_id: a for a in sub.answers}
    for q in questions:
        seg = result.segments.get(q.label.upper())
        ans = existing.get(q.id)
        if ans is None:
            ans = Answer(submission_id=sub.id, question_id=q.id)
            db.session.add(ans)
        elif ans.mapping_confirmed and ans.text_corrected is not None:
            continue  # reviewer-confirmed; never overwrite
        ans_notes = []
        if seg and seg.text:
            ans.text_original, ans.pages = seg.text, seg.pages
            if seg.occurrences > 1:
                ans.status, ans.mapping_confirmed = "ambiguous", False
                ans_notes.append(f"Marker for {q.label} appeared {seg.occurrences} times; confirm the text belongs here.")
            else:
                ans.status, ans.mapping_confirmed = "answered", True
            for pn in seg.pages:
                pt = by_page.get(pn)
                if pt and pt.method != "direct_text" and not pt.reviewed:
                    ans_notes.append(f"Page {pn} was read by OCR and has not been reviewed.")
        elif seg:  # marker present but nothing after it
            ans.text_original, ans.pages = "", seg.pages
            ans.status = "unreadable" if page_errors else "unanswered"
            ans.mapping_confirmed = False
        else:
            ans.text_original, ans.pages = "", []
            ans.status = "unreadable" if page_errors else "unanswered"
            ans.mapping_confirmed = False
            if page_errors:
                ans_notes.append("Some pages failed OCR, so a missing answer cannot be told apart from an unread one.")
        ans.notes = ans_notes
    # No markers at all but exactly one question: offer the text, flagged as ambiguous.
    if result.markers_found == 0 and len(questions) == 1 and pages:
        ans = existing.get(questions[0].id) or Answer.query.filter_by(
            submission_id=sub.id, question_id=questions[0].id).first()
        if ans and not (ans.mapping_confirmed and ans.text_corrected is not None):
            ans.text_original = "\n".join(t for _, t in pages)
            ans.pages = [n for n, _ in pages]
            ans.status, ans.mapping_confirmed = "ambiguous", False
            ans.notes = ["No question number found; this is the whole sheet text - confirm it answers this question."]
    sub.mapping_notes = notes
    sub.status = "mapped"
    db.session.commit()


# ------------------------------------------------------------------ grading
def question_spec(q: Question) -> QuestionSpec:
    return QuestionSpec(
        id=q.id, label=q.label, text=q.text, qtype=q.qtype, max_marks=q.max_marks or 0.0,
        reference_answer=q.reference_answer or "", reference_approved=bool(q.reference_approved),
        criteria=[
            CriterionSpec(id=c.id, name=c.name, max_marks=c.max_marks, description=c.description or "",
                          concepts=[ConceptSpec(id=str(x["id"]), text=x["text"], keywords=x.get("keywords", []),
                                                synonyms=x.get("synonyms", [])) for x in (c.concepts or [])])
            for c in q.criteria
        ],
    )


def build_judge(cfg: dict):
    """Returns (judge, embedder_name). Falls back to the heuristic judge if external is unusable."""
    embedder = get_embedder(cfg["embedder"], cfg["embedder_model"])
    if cfg["judge"] == "external_llm":
        c = current_app.config
        return ExternalLLMJudge(c["EXTERNAL_LLM_URL"], c["EXTERNAL_LLM_KEY"], c["EXTERNAL_LLM_MODEL"]), embedder.name
    return HeuristicJudge(embedder), embedder.name


def evaluate_submission(submission_id: int, progress=None, force: bool = False, judge_override=None):
    sub = db.session.get(Submission, submission_id)
    cfg = settings_svc.get_all()
    try:
        judge, emb_name = (judge_override, "test") if judge_override else build_judge(cfg)
        judge_err = None
    except JudgeError as exc:
        judge, emb_name, judge_err = None, "", str(exc)
    credits = {k[len("credit_"):]: cfg[k] for k in cfg if k.startswith("credit_")}
    engine = GradingEngine(judge, credits, cfg["max_grading_attempts"], emb_name) if judge else None
    questions = Question.query.filter_by(exam_id=sub.exam_id).order_by(Question.position).all()
    map_missing = {a.question_id for a in sub.answers}
    for q in questions:  # make sure every question has an Answer row
        if q.id not in map_missing:
            db.session.add(Answer(submission_id=sub.id, question_id=q.id, status="unanswered"))
    db.session.commit()
    answers = {a.question_id: a for a in sub.answers}
    for i, q in enumerate(questions):
        ans = answers[q.id]
        ev = ans.evaluation or Evaluation(answer_id=ans.id)
        if ans.evaluation is None:
            db.session.add(ev)
        if ev.approved and not force:
            continue
        ev.graded_at = _utcnow()
        if engine is None:
            ev.status, ev.ai_error, ev.ai_total, ev.ai_result = "manual_required", f"Judge unavailable: {judge_err}", None, None
        else:
            outcome = engine.evaluate(question_spec(q), AnswerSpec(ans.effective_text, ans.status, ans.mapping_confirmed))
            flags = list(outcome.flags)
            if outcome.result is not None:
                if ILLEGIBLE in ans.effective_text.lower():
                    flags.append("contains_illegible_text: part of the handwriting was unreadable")
                for note in (ans.notes or []):
                    if "not been reviewed" in note and note not in flags:
                        flags.append(note)
                if flags and outcome.status == "ai_graded":
                    outcome.status = "needs_review"
            ev.status = outcome.status
            ev.ai_error = outcome.error
            ev.ai_flags = flags
            ev.ai_total = outcome.result.total if outcome.result else None
            ev.ai_result = outcome.result.model_dump() if outcome.result else None
            ev.judge_name, ev.embedder_name, ev.attempts = outcome.judge_name, outcome.embedder_name, outcome.attempts
        # a re-grade invalidates any earlier human numbers only if the reviewer forced it
        if force:
            ev.human_criterion_scores = ev.human_total = ev.final_total = None
            ev.approved = False
        db.session.commit()
        if progress:
            progress((i + 1) / len(questions), f"Evaluated {q.label}")
    sub.status = "graded"
    db.session.commit()


# ------------------------------------------------------------------ review
class ReviewError(ValueError):
    pass


def review_evaluation(ev: Evaluation, criterion_scores: Optional[dict], total: Optional[float],
                      comment: Optional[str], approve: bool) -> Evaluation:
    q = ev.answer.question
    crit_by_id = {c.id: c for c in q.criteria}
    max_marks = q.max_marks
    if max_marks is None:
        raise ReviewError("question has no confirmed maximum marks")
    if criterion_scores is not None:
        cleaned = {}
        for key, val in criterion_scores.items():
            try:
                cid, score = int(key), float(val)
            except (TypeError, ValueError):
                raise ReviewError(f"invalid criterion score {key!r}: {val!r}")
            if cid not in crit_by_id:
                raise ReviewError(f"criterion {cid} does not belong to this question")
            if not 0 <= score <= crit_by_id[cid].max_marks:
                raise ReviewError(f"score for '{crit_by_id[cid].name}' must be between 0 and {crit_by_id[cid].max_marks:g}")
            cleaned[str(cid)] = score
        # Criteria the reviewer did not touch keep the AI's awarded marks (if any).
        base = {}
        if ev.ai_result:
            base = {str(c["criterion_id"]): c["awarded"] for c in ev.ai_result["criteria"]}
        merged = {**base, **cleaned}
        ev.human_criterion_scores = cleaned
        ev.human_total = min(sum(merged.values()), max_marks)
        if len(merged) < len(crit_by_id) and not ev.ai_result:
            ev.human_total = None
            raise ReviewError("score every criterion when no AI result exists, or provide a total")
    elif total is not None:
        try:
            total = float(total)
        except (TypeError, ValueError):
            raise ReviewError("total must be a number")
        if not 0 <= total <= max_marks:
            raise ReviewError(f"total must be between 0 and {max_marks:g}")
        ev.human_total = total
        ev.human_criterion_scores = None
    if comment is not None:
        ev.reviewer_comment = comment
    if approve:
        final = ev.human_total if ev.human_total is not None else ev.ai_total
        if final is None:
            raise ReviewError("cannot approve: there is no AI score and no human score")
        if ev.status == "manual_required" and ev.human_total is None:
            raise ReviewError("cannot approve without a human-entered score")
        ev.final_total = min(max(float(final), 0.0), max_marks)
        ev.approved = True
        ev.status = "reviewed"
        ev.reviewed_at = _utcnow()
    else:
        ev.approved, ev.final_total = False, None
        if ev.human_total is not None:
            ev.reviewed_at = _utcnow()
    db.session.commit()
    return ev


# ------------------------------------------------------------------ totals
def main_prefix(label: str) -> str:
    m = re.match(r"Q\d+", label.upper())
    return m.group(0) if m else label.upper()


def _question_marks(ev: Optional[Evaluation]):
    """(marks, kind) where kind is final | human | ai | none."""
    if ev is None:
        return None, "none"
    if ev.approved and ev.final_total is not None:
        return ev.final_total, "final"
    if ev.human_total is not None:
        return ev.human_total, "human"
    if ev.ai_total is not None:
        return ev.ai_total, "ai"
    return None, "none"


def exam_max_marks(exam) -> float:
    qs = [q for q in exam.questions if q.max_marks]
    grouped = {main_prefix(l) for g in (exam.choice_groups or []) for l in g}
    mains: dict[str, float] = {}
    for q in qs:
        mains[main_prefix(q.label)] = mains.get(main_prefix(q.label), 0.0) + q.max_marks
    total = sum(v for k, v in mains.items() if k not in grouped)
    for g in exam.choice_groups or []:
        total += max([mains.get(main_prefix(l), 0.0) for l in g], default=0.0)
    return total


def submission_totals(sub: Submission) -> dict:
    exam = sub.exam
    answers = {a.question_id: a for a in sub.answers}
    rows, mains = [], {}
    for q in exam.questions:
        ans = answers.get(q.id)
        ev = ans.evaluation if ans else None
        marks, kind = _question_marks(ev)
        rows.append({"question_id": q.id, "label": q.label, "max_marks": q.max_marks, "marks": marks, "kind": kind,
                     "status": ev.status if ev else "pending"})
        if marks is not None:
            mains[main_prefix(q.label)] = mains.get(main_prefix(q.label), 0.0) + marks
    grouped = {main_prefix(l) for g in (exam.choice_groups or []) for l in g}
    total = sum(v for k, v in mains.items() if k not in grouped)
    counted = set()
    for g in exam.choice_groups or []:
        best = max(g, key=lambda l: mains.get(main_prefix(l), 0.0), default=None)
        if best is not None:
            total += mains.get(main_prefix(best), 0.0)
            counted.add(main_prefix(best))
    for r in rows:
        mp = main_prefix(r["label"])
        r["counted"] = mp not in grouped or mp in counted
    max_total = exam_max_marks(exam)
    total = min(total, max_total) if max_total else total
    all_final = bool(rows) and all(r["kind"] == "final" for r in rows if r["counted"])
    return {"total": round(total, 2), "max_total": round(max_total, 2),
            "percentage": round(100 * total / max_total, 1) if max_total else None,
            "all_final": all_final, "questions": rows}
