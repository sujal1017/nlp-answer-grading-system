"""JSON representations of models."""
from .models import Answer, Criterion, Document, Evaluation, Exam, Job, PageText, Question, Submission


def iso(dt):
    return dt.isoformat() + "Z" if dt else None


def criterion_dict(c: Criterion):
    return {"id": c.id, "name": c.name, "description": c.description, "max_marks": c.max_marks,
            "concepts": c.concepts or []}


def question_dict(q: Question, with_rubric=True):
    d = {"id": q.id, "exam_id": q.exam_id, "label": q.label, "text": q.text, "context": q.context,
         "max_marks": q.max_marks, "marks_confirmed": q.marks_confirmed, "qtype": q.qtype,
         "source_page": q.source_page, "source": q.source, "needs_review": q.needs_review,
         "review_reasons": q.review_reasons or [], "position": q.position,
         "reference_answer": q.reference_answer, "reference_source": q.reference_source,
         "reference_approved": q.reference_approved}
    if with_rubric:
        d["criteria"] = [criterion_dict(c) for c in q.criteria]
        d["rubric_total"] = round(sum(c.max_marks for c in q.criteria), 4)
    return d


def document_dict(d: Document):
    return {"id": d.id, "kind": d.kind, "filename": d.original_filename, "page_count": d.page_count,
            "size_bytes": d.size_bytes, "pages": d.page_analysis or [], "created_at": iso(d.created_at)}


def exam_dict(e: Exam, detail=False):
    d = {"id": e.id, "name": e.name, "subject": e.subject, "code": e.code, "notes": e.notes,
         "instructions": e.instructions, "choice_groups": e.choice_groups or [],
         "created_at": iso(e.created_at), "question_count": len(e.questions),
         "submission_count": len(e.submissions)}
    if detail:
        d["questions"] = [question_dict(q) for q in e.questions]
        d["documents"] = [document_dict(x) for x in e.documents]
    return d


def page_dict(p: PageText):
    return {"page_number": p.page_number, "kind": p.page_kind, "method": p.method, "engine": p.engine_info,
            "text_original": p.text_original, "text_corrected": p.text_corrected,
            "confidence": p.confidence, "warnings": p.warnings or [], "error": p.error, "reviewed": p.reviewed}


def evaluation_dict(ev: Evaluation):
    if ev is None:
        return None
    return {"id": ev.id, "answer_id": ev.answer_id, "status": ev.status, "ai_total": ev.ai_total,
            "ai_result": ev.ai_result, "ai_error": ev.ai_error, "ai_flags": ev.ai_flags or [],
            "judge": ev.judge_name, "embedder": ev.embedder_name, "attempts": ev.attempts,
            "graded_at": iso(ev.graded_at), "human_criterion_scores": ev.human_criterion_scores,
            "human_total": ev.human_total, "reviewer_comment": ev.reviewer_comment,
            "reviewed_at": iso(ev.reviewed_at), "approved": ev.approved, "final_total": ev.final_total}


def answer_dict(a: Answer):
    return {"id": a.id, "submission_id": a.submission_id, "question_id": a.question_id,
            "question_label": a.question.label, "text_original": a.text_original,
            "text_corrected": a.text_corrected, "pages": a.pages or [], "status": a.status,
            "mapping_confirmed": a.mapping_confirmed, "notes": a.notes or [],
            "evaluation": evaluation_dict(a.evaluation)}


def submission_dict(s: Submission, detail=False):
    d = {"id": s.id, "exam_id": s.exam_id, "exam_name": s.exam.name, "student_identifier": s.student_identifier,
         "status": s.status, "page_count": s.document.page_count, "filename": s.document.original_filename,
         "created_at": iso(s.created_at), "mapping_notes": s.mapping_notes or []}
    if detail:
        d["pages"] = [page_dict(p) for p in s.pages]
        d["answers"] = [answer_dict(a) for a in sorted(s.answers, key=lambda a: a.question.position)]
        d["unmapped_text"] = s.unmapped_text
    return d


def job_dict(j: Job):
    return {"id": j.id, "kind": j.kind, "ref_id": j.ref_id, "status": j.status, "progress": j.progress,
            "message": j.message, "error": j.error}
