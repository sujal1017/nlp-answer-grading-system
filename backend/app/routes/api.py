"""REST API. Every endpoint talks to the real services - there is no mock data path."""
import io
import shutil
from datetime import timedelta

import pymupdf as fitz
from flask import Blueprint, Response, current_app, jsonify, request, send_file

from ..extensions import db
from ..grading.embeddings import EmbedderUnavailable
from ..grading.judges import ExternalLLMJudge, JudgeError
from ..models import (Answer, Criterion, Document, Evaluation, Exam, Job, Question, Submission, _now)
from ..serializers import (answer_dict, document_dict, evaluation_dict, exam_dict, job_dict, page_dict,
                           question_dict, submission_dict)
from ..services import metrics as metrics_svc
from ..services import pdf_processing as pdfp
from ..services import reports, workflow
from ..services import settings as settings_svc
from ..services.jobs import create_job, run_job
from ..services.question_extraction import detect_qtype
from ..services.storage import UploadError, resolve_stored, safe_display_name, save_pdf_bytes, validate_pdf_upload

bp = Blueprint("api", __name__, url_prefix="/api")
QTYPES = {"definition", "explanation", "comparison", "application", "numerical", "algorithm", "long_answer", "unknown"}


def err(message, status=400, **extra):
    return jsonify({"error": message, **extra}), status


def body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def get_or_404(model, ident):
    obj = db.session.get(model, ident)
    if obj is None:
        raise NotFound(f"{model.__name__.lower()} {ident} not found")
    return obj


class NotFound(Exception):
    pass


@bp.errorhandler(NotFound)
def _nf(e):
    return err(str(e), 404)


@bp.errorhandler(UploadError)
def _ue(e):
    return err(str(e), 400)


@bp.errorhandler(pdfp.PDFError)
def _pe(e):
    return err(str(e), 400)


@bp.errorhandler(workflow.ReviewError)
def _re(e):
    db.session.rollback()
    return err(str(e), 422)


# ------------------------------------------------------------------ system
@bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@bp.get("/system")
def system():
    import importlib.util as iu
    import shutil as sh

    def has(m):
        return iu.find_spec(m) is not None
    c = current_app.config
    return jsonify({
        "tesseract": sh.which("tesseract") is not None, "torch": has("torch"), "transformers": has("transformers"),
        "sentence_transformers": has("sentence_transformers"),
        "external_llm_configured": bool(c["EXTERNAL_LLM_URL"] and c["EXTERNAL_LLM_MODEL"]),
        "max_upload_mb": c["MAX_UPLOAD_MB"],
        "notes": ["Qwen2-VL OCR needs torch + transformers and a model download.",
                  "Without sentence-transformers the lexical fallback embedder is used (not semantic)."],
    })


@bp.get("/settings")
def get_settings():
    return jsonify({"settings": settings_svc.get_all(), "choices": settings_svc.SETTING_CHOICES})


@bp.put("/settings")
def put_settings():
    try:
        return jsonify({"settings": settings_svc.validate_and_set(body())})
    except ValueError as exc:
        return err("invalid settings", 422, details=exc.args[0])


@bp.get("/jobs/<int:job_id>")
def get_job(job_id):
    return jsonify(job_dict(get_or_404(Job, job_id)))


# ------------------------------------------------------------------ exams
@bp.get("/exams")
def list_exams():
    return jsonify([exam_dict(e) for e in Exam.query.order_by(Exam.created_at.desc()).all()])


@bp.post("/exams")
def create_exam():
    d = body()
    name = (d.get("name") or "").strip()
    if not name:
        return err("name is required", 422)
    e = Exam(name=name[:200], subject=(d.get("subject") or "")[:200], code=(d.get("code") or "")[:60],
             notes=d.get("notes") or "", choice_groups=_clean_groups(d.get("choice_groups")))
    db.session.add(e)
    db.session.commit()
    return jsonify(exam_dict(e, True)), 201


def _clean_groups(groups):
    if not groups:
        return []
    if not (isinstance(groups, list) and all(isinstance(g, list) and all(isinstance(x, str) for x in g) for g in groups)):
        raise UploadError("choice_groups must be a list of lists of question labels like [['Q1','Q2']]")
    return [[x.strip().upper() for x in g if x.strip()] for g in groups if g]


@bp.get("/exams/<int:exam_id>")
def get_exam(exam_id):
    e = get_or_404(Exam, exam_id)
    d = exam_dict(e, True)
    d["max_marks"] = workflow.exam_max_marks(e)
    return jsonify(d)


@bp.patch("/exams/<int:exam_id>")
def patch_exam(exam_id):
    e, d = get_or_404(Exam, exam_id), body()
    for f, n in (("name", 200), ("subject", 200), ("code", 60)):
        if f in d:
            if f == "name" and not str(d[f]).strip():
                return err("name cannot be empty", 422)
            setattr(e, f, str(d[f])[:n])
    if "notes" in d:
        e.notes = d["notes"] or ""
    if "choice_groups" in d:
        e.choice_groups = _clean_groups(d["choice_groups"])
    db.session.commit()
    return jsonify(exam_dict(e, True))


@bp.delete("/exams/<int:exam_id>")
def delete_exam(exam_id):
    e = get_or_404(Exam, exam_id)
    _delete_files([x.stored_name for x in e.documents])
    db.session.delete(e)
    db.session.commit()
    return "", 204


def _delete_files(names):
    for n in names:
        try:
            resolve_stored(n).unlink(missing_ok=True)
        except Exception:
            current_app.logger.warning("could not delete stored file %s", n)


# ------------------------------------------------------------------ question paper
@bp.post("/exams/<int:exam_id>/question-paper")
def upload_question_paper(exam_id):
    exam = get_or_404(Exam, exam_id)
    data = validate_pdf_upload(request.files.get("file"))
    cfg = settings_svc.get_all()
    pages = pdfp.analyze_pdf(data, cfg["max_pages"])  # raises PDFError for corrupt/encrypted/too long
    stored, sha = save_pdf_bytes(data)
    doc = Document(exam_id=exam.id, kind="question_paper", original_filename=safe_display_name(request.files["file"].filename),
                   stored_name=stored, sha256=sha, size_bytes=len(data), page_count=len(pages),
                   page_analysis=[p.to_dict() for p in pages])
    db.session.add(doc)
    db.session.commit()
    doc_id = doc.id  # capture now: the request session is closed by the time a background thread runs
    job = create_job("extract_questions", doc_id)
    result = {}

    def work(progress):
        result.update(workflow.extract_question_paper(db.session.get(Exam, exam_id), db.session.get(Document, doc_id), progress))

    run_job(job, work)
    db.session.refresh(job)
    payload = {"document": document_dict(doc), "job": job_dict(job), "result": result or None}
    if job.status == "done":
        db.session.refresh(exam)
        payload["questions"] = [question_dict(q) for q in exam.questions]
    return jsonify(payload), 201 if job.status == "done" else 202


@bp.get("/documents/<int:doc_id>/pages/<int:page>/image")
def document_page_image(doc_id, page):
    doc = get_or_404(Document, doc_id)
    return _page_png(doc, page)


def _page_png(doc: Document, page: int):
    if not 1 <= page <= doc.page_count:
        raise NotFound("page out of range")
    dpi = max(50, min(int(request.args.get("dpi", 110)), 220))
    pdf = pdfp.open_pdf(resolve_stored(doc.stored_name).read_bytes())
    try:
        png = pdfp.render_page_png(pdf, page - 1, dpi)
    finally:
        pdf.close()
    return send_file(io.BytesIO(png), mimetype="image/png", max_age=300)


# ------------------------------------------------------------------ questions & rubric
@bp.get("/exams/<int:exam_id>/questions")
def list_questions(exam_id):
    return jsonify([question_dict(q) for q in get_or_404(Exam, exam_id).questions])


@bp.post("/exams/<int:exam_id>/questions")
def add_question(exam_id):
    exam, d = get_or_404(Exam, exam_id), body()
    label, text = (d.get("label") or "").strip().upper(), (d.get("text") or "").strip()
    if not label or not text:
        return err("label and text are required", 422)
    if any(q.label == label for q in exam.questions):
        return err(f"a question labelled {label} already exists", 409)
    marks = _parse_marks(d.get("max_marks"))
    q = Question(exam_id=exam.id, label=label, text=text, max_marks=marks, marks_confirmed=marks is not None,
                 qtype=d.get("qtype") if d.get("qtype") in QTYPES else detect_qtype(text, marks),
                 position=max([x.position for x in exam.questions], default=-1) + 1, source="manual",
                 needs_review=marks is None,
                 review_reasons=[] if marks is not None else ["Marks not provided."])
    db.session.add(q)
    db.session.commit()
    return jsonify(question_dict(q)), 201


def _parse_marks(v):
    if v in (None, ""):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise UploadError("max_marks must be a number")
    if f <= 0 or f > 1000:
        raise UploadError("max_marks must be greater than 0")
    return f


@bp.patch("/questions/<int:qid>")
def patch_question(qid):
    q, d = get_or_404(Question, qid), body()
    if "text" in d:
        if not str(d["text"]).strip():
            return err("text cannot be empty", 422)
        q.text = str(d["text"]).strip()
    if "label" in d:
        label = str(d["label"]).strip().upper()
        if label != q.label and any(x.label == label for x in q.exam.questions):
            return err(f"a question labelled {label} already exists", 409)
        q.label = label or q.label
    if "qtype" in d:
        if d["qtype"] not in QTYPES:
            return err(f"qtype must be one of {sorted(QTYPES)}", 422)
        q.qtype = d["qtype"]
    if "max_marks" in d:
        marks = _parse_marks(d["max_marks"])
        if marks is not None and q.criteria and abs(sum(c.max_marks for c in q.criteria) - marks) > 1e-6:
            q.review_reasons = (q.review_reasons or []) + ["Marks changed; rubric criteria no longer add up."]
        q.max_marks, q.marks_confirmed = marks, marks is not None
    q.needs_review = q.max_marks is None
    q.review_reasons = [] if q.max_marks is not None else ["Marks not confirmed."]
    db.session.commit()
    return jsonify(question_dict(q))


@bp.delete("/questions/<int:qid>")
def delete_question(qid):
    q = get_or_404(Question, qid)
    if Answer.query.filter_by(question_id=q.id).first():
        return err("this question has student answers; delete those submissions first", 409)
    db.session.delete(q)
    db.session.commit()
    return "", 204


@bp.put("/questions/<int:qid>/reference")
def put_reference(qid):
    q, d = get_or_404(Question, qid), body()
    text = (d.get("text") or "").strip()
    if not text:
        return err("reference answer text is required", 422)
    q.reference_answer, q.reference_source, q.reference_approved = text, "human", True
    db.session.commit()
    return jsonify(question_dict(q))


@bp.post("/questions/<int:qid>/reference/draft")
def draft_reference(qid):
    q = get_or_404(Question, qid)
    c = current_app.config
    if not (c["EXTERNAL_LLM_URL"] and c["EXTERNAL_LLM_MODEL"]):
        return err("AI drafting needs an external LLM (set EXTERNAL_LLM_URL / EXTERNAL_LLM_MODEL). "
                   "Enter the reference answer manually instead.", 503)
    try:
        text = ExternalLLMJudge(c["EXTERNAL_LLM_URL"], c["EXTERNAL_LLM_KEY"], c["EXTERNAL_LLM_MODEL"]) \
            .draft_reference_answer(q.text, q.max_marks)
    except JudgeError as exc:
        return err(str(exc), 502)
    q.reference_answer, q.reference_source, q.reference_approved = text, "ai_draft", False
    db.session.commit()
    return jsonify(question_dict(q))


@bp.post("/questions/<int:qid>/reference/approve")
def approve_reference(qid):
    q = get_or_404(Question, qid)
    if not q.reference_answer.strip():
        return err("there is no reference answer to approve", 422)
    q.reference_approved = True
    db.session.commit()
    return jsonify(question_dict(q))


@bp.put("/questions/<int:qid>/rubric")
def put_rubric(qid):
    q, d = get_or_404(Question, qid), body()
    crits = d.get("criteria")
    if not isinstance(crits, list) or not crits:
        return err("criteria must be a non-empty list", 422)
    problems, parsed, n = [], [], 0
    for i, c in enumerate(crits):
        name = str((c or {}).get("name") or "").strip()
        try:
            mm = float((c or {}).get("max_marks"))
        except (TypeError, ValueError):
            mm = -1
        if not name:
            problems.append(f"criterion {i + 1}: name is required")
        if mm < 0:
            problems.append(f"criterion {i + 1}: max_marks must be a number >= 0")
        concepts = []
        for x in (c or {}).get("concepts") or []:
            t = str((x or {}).get("text") or "").strip()
            if not t:
                continue
            n += 1
            concepts.append({"id": f"c{n}", "text": t,
                             "keywords": [str(k).strip() for k in x.get("keywords", []) if str(k).strip()],
                             "synonyms": [str(k).strip() for k in x.get("synonyms", []) if str(k).strip()]})
        if not concepts:
            problems.append(f"criterion {i + 1} ('{name}'): add at least one expected concept")
        parsed.append((name, str(c.get("description") or ""), mm, concepts))
    if q.max_marks is None:
        problems.append("confirm the question's maximum marks before defining a rubric")
    else:
        total = round(sum(p[2] for p in parsed if p[2] >= 0), 4)
        if abs(total - q.max_marks) > 1e-6:
            problems.append(f"criteria add up to {total:g} but the question is worth {q.max_marks:g}")
    if problems:
        return err("invalid rubric", 422, details=problems)
    approved = [a for a in Answer.query.filter_by(question_id=q.id).all() if a.evaluation and a.evaluation.approved]
    if approved and not request.args.get("force"):
        return err("approved evaluations exist for this question; changing the rubric would invalidate them. "
                   "Pass ?force=1 to discard those evaluations.", 409)
    for a in Answer.query.filter_by(question_id=q.id).all():
        if a.evaluation:
            db.session.delete(a.evaluation)
    for old in list(q.criteria):
        db.session.delete(old)
    db.session.flush()
    for pos, (name, desc, mm, concepts) in enumerate(parsed):
        db.session.add(Criterion(question_id=q.id, position=pos, name=name[:200], description=desc, max_marks=mm,
                                 concepts=concepts))
    db.session.commit()
    db.session.refresh(q)
    return jsonify(question_dict(q))


# ------------------------------------------------------------------ answer sheets
@bp.post("/exams/<int:exam_id>/submissions")
def upload_answer_sheet(exam_id):
    exam = get_or_404(Exam, exam_id)
    if not exam.questions:
        return err("add questions to this exam before uploading answer sheets", 409)
    data = validate_pdf_upload(request.files.get("file"))
    cfg = settings_svc.get_all()
    pages = pdfp.analyze_pdf(data, cfg["max_pages"])
    stored, sha = save_pdf_bytes(data)
    doc = Document(exam_id=exam.id, kind="answer_sheet", original_filename=safe_display_name(request.files["file"].filename),
                   stored_name=stored, sha256=sha, size_bytes=len(data), page_count=len(pages),
                   page_analysis=[p.to_dict() for p in pages])
    db.session.add(doc)
    db.session.flush()
    sub = Submission(exam_id=exam.id, document_id=doc.id,
                     student_identifier=(request.form.get("student_identifier") or "").strip()[:100])
    db.session.add(sub)
    db.session.commit()
    job = create_job("answer_sheet_ocr", sub.id)
    sub_id = sub.id  # capture now: the request session is closed by the time the thread runs
    run_job(job, lambda progress: workflow.run_answer_sheet_ocr(sub_id, progress))
    db.session.refresh(job)
    db.session.refresh(sub)
    return jsonify({"submission": submission_dict(sub, True), "job": job_dict(job)}), 202 if job.status != "done" else 201


@bp.get("/submissions")
def list_submissions():
    q = Submission.query.order_by(Submission.created_at.desc())
    if request.args.get("exam_id"):
        q = q.filter_by(exam_id=int(request.args["exam_id"]))
    if request.args.get("status"):
        q = q.filter_by(status=request.args["status"])
    out = []
    for s in q.limit(200).all():
        d = submission_dict(s)
        t = workflow.submission_totals(s)
        d.update({"total": t["total"], "max_total": t["max_total"], "percentage": t["percentage"],
                  "all_final": t["all_final"],
                  "needs_review": sum(1 for a in s.answers if a.evaluation and a.evaluation.status in
                                      ("needs_review", "manual_required"))})
        out.append(d)
    return jsonify(out)


@bp.get("/submissions/<int:sid>")
def get_submission(sid):
    s = get_or_404(Submission, sid)
    d = submission_dict(s, True)
    d["totals"] = workflow.submission_totals(s)
    d["questions"] = [question_dict(q) for q in s.exam.questions]
    return jsonify(d)


@bp.delete("/submissions/<int:sid>")
def delete_submission(sid):
    s = get_or_404(Submission, sid)
    doc = s.document
    name = doc.stored_name
    db.session.delete(s)
    db.session.flush()
    db.session.delete(doc)
    db.session.commit()
    _delete_files([name])
    return "", 204


@bp.get("/submissions/<int:sid>/pages/<int:page>/image")
def submission_page_image(sid, page):
    return _page_png(get_or_404(Submission, sid).document, page)


@bp.patch("/submissions/<int:sid>")
def patch_submission(sid):
    s, d = get_or_404(Submission, sid), body()
    if "student_identifier" in d:
        s.student_identifier = str(d["student_identifier"]).strip()[:100]
    db.session.commit()
    return jsonify(submission_dict(s, True))


@bp.post("/submissions/<int:sid>/ocr")
def rerun_ocr(sid):
    s = get_or_404(Submission, sid)
    job = create_job("answer_sheet_ocr", s.id)
    sub_id = s.id
    run_job(job, lambda progress: workflow.run_answer_sheet_ocr(sub_id, progress))
    db.session.refresh(job)
    return jsonify({"job": job_dict(job)}), 202


@bp.patch("/submissions/<int:sid>/pages/<int:page>")
def patch_page(sid, page):
    s, d = get_or_404(Submission, sid), body()
    p = next((x for x in s.pages if x.page_number == page), None)
    if p is None:
        raise NotFound("page not found")
    if "text_corrected" in d:
        p.text_corrected = d["text_corrected"] if d["text_corrected"] is not None else None
    if "reviewed" in d:
        p.reviewed = bool(d["reviewed"])
    db.session.commit()
    return jsonify(page_dict(p))


@bp.post("/submissions/<int:sid>/map")
def remap(sid):
    s = get_or_404(Submission, sid)
    workflow.map_submission(s)
    db.session.refresh(s)
    return jsonify(submission_dict(s, True))


@bp.patch("/answers/<int:aid>")
def patch_answer(aid):
    a, d = get_or_404(Answer, aid), body()
    if "question_id" in d and d["question_id"] != a.question_id:
        target = db.session.get(Question, d["question_id"])
        if target is None or target.exam_id != a.submission.exam_id:
            return err("target question does not belong to this exam", 422)
        clash = Answer.query.filter_by(submission_id=a.submission_id, question_id=target.id).first()
        if clash and clash.effective_text.strip():
            return err(f"{target.label} already has an answer; clear it first", 409)
        if clash:
            db.session.delete(clash)
            db.session.flush()
        a.question_id = target.id
        if a.evaluation:
            db.session.delete(a.evaluation)
    if "text_corrected" in d:
        a.text_corrected = d["text_corrected"]
    if "status" in d:
        if d["status"] not in ("answered", "unanswered", "unreadable", "ambiguous"):
            return err("invalid status", 422)
        a.status = d["status"]
    if "mapping_confirmed" in d:
        a.mapping_confirmed = bool(d["mapping_confirmed"])
        if a.mapping_confirmed and a.status == "ambiguous":
            a.status = "answered"
    if a.effective_text.strip() and a.status in ("unanswered", "unreadable") and "status" not in d:
        a.status = "answered"
    db.session.commit()
    db.session.refresh(a)
    return jsonify(answer_dict(a))


# ------------------------------------------------------------------ evaluation
@bp.post("/submissions/<int:sid>/evaluate")
def evaluate(sid):
    s = get_or_404(Submission, sid)
    force = bool(body().get("force"))
    job = create_job("evaluate", s.id)
    sub_id = s.id
    run_job(job, lambda progress: workflow.evaluate_submission(sub_id, progress, force))
    db.session.refresh(job)
    return jsonify({"job": job_dict(job)}), 202


@bp.post("/answers/<int:aid>/manual-evaluation")
def manual_evaluation(aid):
    a = get_or_404(Answer, aid)
    if a.evaluation is None:
        db.session.add(Evaluation(answer_id=a.id, status="manual_required", ai_error="Manual grading requested."))
        db.session.commit()
        db.session.refresh(a)
    return jsonify(evaluation_dict(a.evaluation)), 201


@bp.patch("/evaluations/<int:eid>/review")
def review(eid):
    ev, d = get_or_404(Evaluation, eid), body()
    workflow.review_evaluation(ev, d.get("criterion_scores"), d.get("total"), d.get("comment"), bool(d.get("approve")))
    sub = ev.answer.submission
    if all(a.evaluation and a.evaluation.approved for a in sub.answers) and sub.answers:
        sub.status = "reviewed"
        db.session.commit()
    return jsonify(evaluation_dict(ev))


# ------------------------------------------------------------------ results & reports
@bp.get("/submissions/<int:sid>/results")
def results(sid):
    return jsonify(workflow.submission_totals(get_or_404(Submission, sid)))


@bp.get("/submissions/<int:sid>/report.pdf")
def report_pdf(sid):
    s = get_or_404(Submission, sid)
    return send_file(io.BytesIO(reports.submission_pdf(s)), mimetype="application/pdf", as_attachment=True,
                     download_name=f"report_{s.id}.pdf")


@bp.get("/submissions/<int:sid>/report.csv")
def report_csv(sid):
    s = get_or_404(Submission, sid)
    return Response(reports.submission_csv(s), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=report_{s.id}.csv"})


@bp.get("/exams/<int:exam_id>/report.csv")
def exam_report_csv(exam_id):
    e = get_or_404(Exam, exam_id)
    return Response(reports.exam_csv(e), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=exam_{e.id}.csv"})


@bp.get("/exams/<int:exam_id>/metrics")
def exam_metrics(exam_id):
    e = get_or_404(Exam, exam_id)
    cfg = settings_svc.get_all()
    ai, human = [], []
    for s in e.submissions:
        for a in s.answers:
            ev = a.evaluation
            if ev and ev.ai_total is not None and (ev.human_total is not None or (ev.approved and ev.final_total is not None)):
                ai.append(ev.ai_total)
                human.append(ev.human_total if ev.human_total is not None else ev.final_total)
    tol = float(request.args.get("tolerance", cfg["mark_tolerance"]))
    return jsonify(metrics_svc.compute_metrics(ai, human, tol, cfg["min_metric_samples"]))


# ------------------------------------------------------------------ dashboard & maintenance
@bp.get("/dashboard")
def dashboard():
    subs = Submission.query.order_by(Submission.created_at.desc()).all()
    need_review = [s for s in subs if any(a.evaluation and not a.evaluation.approved and
                                          a.evaluation.status in ("needs_review", "manual_required")
                                          for a in s.answers)]
    running = Job.query.filter(Job.status.in_(["queued", "running"])).count()
    return jsonify({
        "exam_count": Exam.query.count(), "submission_count": len(subs),
        "processing": sum(1 for s in subs if s.status in ("uploaded", "ocr_running")) + running,
        "completed": sum(1 for s in subs if s.status == "reviewed"),
        "needs_review": len(need_review),
        "recent": [{**submission_dict(s), **{k: v for k, v in workflow.submission_totals(s).items() if k in
                    ("total", "max_total", "percentage", "all_final")}} for s in subs[:8]],
        "review_queue": [submission_dict(s) for s in need_review[:8]],
    })


@bp.post("/maintenance/purge")
def purge():
    """Delete submissions older than the configured retention period (0 = keep forever)."""
    days = int(settings_svc.get("retention_days"))
    if days <= 0:
        return jsonify({"deleted": 0, "message": "Retention is disabled (keep forever)."})
    cutoff = _now() - timedelta(days=days)
    old = Submission.query.filter(Submission.created_at < cutoff).all()
    n = 0
    for s in old:
        name, doc = s.document.stored_name, s.document
        db.session.delete(s)
        db.session.flush()
        db.session.delete(doc)
        _delete_files([name])
        n += 1
    db.session.commit()
    return jsonify({"deleted": n})
