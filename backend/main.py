import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chunker import (
    calculate_ia_total,
    chunk_student_booklet,
    convert_pdf_to_images,
    parse_teacher_questions,
)
from database import (
    BACKEND_DIR,
    DATA_DIR,
    assign_segment_to_question,
    create_exam,
    delete_all_student_submissions,
    delete_exam_rubrics,
    delete_student_submission,
    get_all_exams,
    get_exam,
    get_exam_booklets,
    get_exam_rubrics,
    get_exam_submissions,
    get_kpi_counts,
    get_raw_booklet,
    init_db,
    merge_and_reassign_submissions,
    save_question_rubric,
    save_raw_booklet,
    save_student_answer,
    split_at_cursor,
    split_submission,
    update_evaluation,
    update_rubric_weights_from_feedback,
    verify_submission,
)
from engine import LocalGradingEngine, generate_atomic_rubric

init_db()
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="EvalAI — Offline University Exam Evaluation", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=str(DATA_DIR)), name="files")

engine = LocalGradingEngine()
jobs: Dict[str, Dict[str, Any]] = {}
grade_lock = threading.Lock()

ROOT_TEST_DIR = BACKEND_DIR.parent / "test"


class QuestionIn(BaseModel):
    question_id: str
    question_text: str
    max_marks: int = Field(default=5, gt=0)
    key_points: Optional[List[Any]] = None
    reference_answer: str = ""


class ExamIn(BaseModel):
    exam_id: str
    exam_title: str
    subject: str = "Research Methodology & IPR"
    questions: List[QuestionIn] = Field(default_factory=list)
    replace_rubrics: bool = True


class ParseIn(BaseModel):
    text: str


class RubricUpdate(BaseModel):
    question_text: str
    max_marks: int = Field(gt=0)
    key_points: List[Any]
    reference_answer: str = ""


class VerifyIn(BaseModel):
    exam_id: str
    student_id: str
    question_id: str
    teacher_score: float
    feedback: str = ""


class SampleGradeIn(BaseModel):
    student_id: str = "2SD23CS082"
    sample_name: str = "test1.pdf"
    max_pages: Optional[int] = None


class MergeReassignRequest(BaseModel):
    exam_id: str
    student_id: str
    source_question_ids: List[str]
    target_question_id: str


class SplitRequest(BaseModel):
    exam_id: str
    student_id: str
    source_question_id: str
    target_question_id: str
    split_index: int
    current_text: Optional[str] = None


class EvaluateStudentRequest(BaseModel):
    exam_id: str
    student_id: str
    question_id: Optional[str] = None


class AssignSegmentRequest(BaseModel):
    exam_id: str
    student_id: str
    question_id: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    custom_text: Optional[str] = None
    remove_from_unassigned: bool = True


class SplitAtCursorRequest(BaseModel):
    exam_id: str
    student_id: str
    source_question_id: str
    cursor_index: int
    target_question_id: str


def public_file_url(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(DATA_DIR.resolve())
            return "/files/" + rel.as_posix()
        except Exception:
            pass
    norm = p.as_posix().lstrip("/\\")
    return "/files/" + norm


async def _store_uploads(exam_id: str, student_id: str, files: List[UploadFile]) -> List[str]:
    student_dir = DATA_DIR / "booklets" / exam_id / student_id
    student_dir.mkdir(parents=True, exist_ok=True)
    page_paths = []
    for file_index, upload in enumerate(files, start=1):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
            raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")
        raw = await upload.read()
        if not raw:
            raise HTTPException(400, f"Uploaded file is empty: {upload.filename or file_index}")

        upload_dir = student_dir / f"upload_{file_index:03d}"
        if suffix == ".pdf":
            page_paths.extend(convert_pdf_to_images(raw, str(upload_dir)))
        else:
            upload_dir.mkdir(parents=True, exist_ok=True)
            image_path = upload_dir / f"page_001{suffix}"
            image_path.write_bytes(raw)
            page_paths.append(str(image_path))

    if not page_paths:
        raise HTTPException(400, "No pages were found in the uploaded files.")
    return page_paths


def _relative_page_paths(page_paths: List[str]) -> List[str]:
    relative_paths = []
    for page_path in page_paths:
        try:
            relative_paths.append(
                Path(page_path).resolve().relative_to(DATA_DIR.resolve()).as_posix()
            )
        except ValueError:
            relative_paths.append(page_path)
    return relative_paths


def _transcribe_and_grade(
    exam_id: str,
    student_id: str,
    page_paths: List[str],
    progress_cb=None,
) -> Dict[str, Any]:
    def report(message: str):
        if progress_cb:
            progress_cb(message)

    report("OCR: starting TrOCR handwriting recognition")
    transcript = engine.transcribe_images(page_paths, progress_cb=progress_cb, auto_unload=True)
    relative_paths = _relative_page_paths(page_paths)
    save_raw_booklet(exam_id, student_id, relative_paths, transcript)

    if not transcript.strip():
        delete_all_student_submissions(exam_id, student_id)
        return {
            "transcript": transcript,
            "submissions": [],
            "chunks": [],
            "ia": calculate_ia_total({}),
            "needs_review": True,
            "message": "OCR found no readable text. Check the uploaded pages and handwriting.",
        }

    rubrics = get_exam_rubrics(exam_id)
    rubric_map = {rubric["question_id"].upper(): rubric for rubric in rubrics}
    chunks = chunk_student_booklet(transcript, rubrics)
    grade_inputs = [
        (question_id.upper(), answer.strip())
        for question_id, answer in chunks.items()
        if question_id.upper() in rubric_map and answer.strip()
    ]

    if not grade_inputs:
        delete_all_student_submissions(exam_id, student_id)
        return {
            "transcript": transcript,
            "submissions": [],
            "chunks": [],
            "ia": calculate_ia_total({}),
            "needs_review": True,
            "message": "OCR is complete, but no question sections matched the exam rubrics. Review and assign the extracted text.",
        }

    evaluated = []
    try:
        for index, (question_id, answer) in enumerate(grade_inputs, start=1):
            rubric = rubric_map[question_id]
            report(f"BERT: grading {question_id} ({index}/{len(grade_inputs)})")
            result = engine.evaluate_answer(
                question_text=rubric["question_text"],
                max_marks=rubric["max_marks"],
                key_points=rubric["key_points"],
                student_answer=answer,
                reference_answer=rubric.get("reference_answer", ""),
            )
            evaluated.append((question_id, answer, result))

        delete_all_student_submissions(exam_id, student_id)
        scores = {}
        for question_id, answer, result in evaluated:
            save_student_answer(exam_id, student_id, question_id, relative_paths, answer)
            update_evaluation(
                exam_id=exam_id,
                student_id=student_id,
                question_id=question_id,
                matched_points=result["matched_points"],
                missing_points=result["missing_points"],
                ai_score=result["ai_score"],
                ai_feedback=result["ai_feedback"],
                similarity_score=result["similarity_score"],
                keyword_coverage=result["keyword_coverage"],
            )
            scores[question_id] = result["ai_score"]
    finally:
        engine.unload_model()

    submissions = get_exam_submissions(exam_id, student_id)
    for submission in submissions:
        submission["max_marks"] = rubric_map[submission["question_id"].upper()]["max_marks"]
    return {
        "transcript": transcript,
        "submissions": submissions,
        "chunks": [question_id for question_id, _, _ in evaluated],
        "ia": calculate_ia_total(scores),
        "needs_review": False,
        "message": "TrOCR extraction and BERT grading complete.",
    }


@app.get("/api/health")
def health():
    return {"ok": True, **engine.telemetry()}


@app.get("/api/telemetry")
def telemetry():
    return engine.telemetry()


@app.post("/api/engine/unload")
def unload_engine():
    engine.unload_model()
    return {"ok": True, **engine.telemetry()}


@app.get("/api/exams")
def list_exams():
    return get_all_exams()


@app.post("/api/parse-questions")
def parse_questions(payload: ParseIn):
    parsed = parse_teacher_questions(payload.text)
    for q in parsed:
        q["key_points"] = generate_atomic_rubric(q["question_text"], q["max_marks"])
    return {"questions": parsed}


@app.post("/api/exams")
def upsert_exam(payload: ExamIn):
    exam_id = payload.exam_id.strip().upper()
    if not exam_id:
        raise HTTPException(400, "Exam ID is required")
    if not payload.questions:
        raise HTTPException(400, "Add at least one question with maximum marks")

    raw_text = "\n".join(
        f"{q.question_id} {q.question_text} {q.max_marks} Marks" for q in payload.questions
    )
    create_exam(exam_id, payload.exam_title, [], raw_text)
    if payload.replace_rubrics:
        delete_exam_rubrics(exam_id)

    saved = []
    for q in payload.questions:
        qid = q.question_id.strip().upper().replace(" ", "")
        points = q.key_points or generate_atomic_rubric(q.question_text, q.max_marks)
        save_question_rubric(
            exam_id,
            qid,
            q.question_text.strip(),
            int(q.max_marks),
            points,
            q.reference_answer,
        )
        saved.append({
            "question_id": qid,
            "question_text": q.question_text,
            "max_marks": q.max_marks,
            "key_points": points,
            "reference_answer": q.reference_answer,
        })
    return {"exam_id": exam_id, "questions": saved}


@app.get("/api/exams/{exam_id}")
def exam_detail(exam_id: str):
    exam = get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    rubrics = get_exam_rubrics(exam_id)
    kpis = get_kpi_counts(exam_id)
    return {"exam": exam, "rubrics": rubrics, "kpis": kpis}


@app.get("/api/exams/{exam_id}/rubrics")
def exam_rubrics(exam_id: str):
    return get_exam_rubrics(exam_id)


@app.put("/api/exams/{exam_id}/rubrics/{question_id}")
def update_rubric(exam_id: str, question_id: str, payload: RubricUpdate):
    save_question_rubric(
        exam_id,
        question_id,
        payload.question_text,
        payload.max_marks,
        payload.key_points,
        payload.reference_answer,
    )
    return {"ok": True}


@app.get("/api/exams/{exam_id}/kpis")
def exam_kpis(exam_id: str):
    return get_kpi_counts(exam_id)


@app.get("/api/exams/{exam_id}/submissions")
def exam_submissions(exam_id: str, student_id: Optional[str] = None):
    rows = get_exam_submissions(exam_id, student_id)
    for row in rows:
        row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]

    # Include students from both submissions and digitized booklets
    booklets = get_exam_booklets(exam_id)
    all_student_ids = set([r["student_id"] for r in rows] + [b["student_id"] for b in booklets])
    students = sorted(all_student_ids)

    return {"students": students, "submissions": rows}


@app.get("/api/booklets/{exam_id}/{student_id}")
def get_booklet_detail(exam_id: str, student_id: str):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    raw_booklet = get_raw_booklet(eid, sid)
    subs = get_exam_submissions(eid, sid)
    for row in subs:
        row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in subs}
    ia_breakdown = calculate_ia_total(scores)

    booklet_dict = None
    if raw_booklet:
        booklet_dict = {
            **raw_booklet,
            "image_urls": [public_file_url(p) for p in raw_booklet.get("image_paths", [])],
        }

    return {
        "booklet": booklet_dict,
        "submissions": subs,
        "ia_total": ia_breakdown["ia_total"],
        "ia_breakdown": ia_breakdown,
    }


@app.post("/api/submissions/upload")
async def upload_booklet(
    exam_id: str = Form(...),
    student_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    if not eid or not sid:
        raise HTTPException(400, "exam_id and student_id are required")
    if not files:
        raise HTTPException(400, "Upload at least one booklet PDF or image")

    page_paths = await _store_uploads(eid, sid, files)
    result = _transcribe_and_grade(eid, sid, page_paths)
    return {
        "status": "needs_review" if result["needs_review"] else "evaluated",
        "student_id": sid,
        "pages": len(page_paths),
        "text": result["transcript"],
        "submissions": result["submissions"],
        "chunks": result["chunks"],
        "ia": result["ia"],
        "message": result["message"],
        "image_urls": [public_file_url(p) for p in _relative_page_paths(page_paths)],
    }


@app.post("/api/submissions/split")
def split_submission_endpoint(payload: SplitRequest):
    """
    Step 2: Cursor Splitting & Question Assignment in Tab 3:
      - split_index == 0: Moves entire text of source_question_id to target_question_id,
        and deletes the empty source_question_id card!
      - split_index > 0: Keeps text[:split_index] in source_question_id and moves
        text[split_index:] to target_question_id.
    """
    eid = payload.exam_id.strip().upper()
    sid = payload.student_id.strip().upper()
    source_qid = payload.source_question_id.strip().upper()
    target_qid = payload.target_question_id.strip().upper()

    if payload.split_index is None or payload.split_index < 0:
        raise HTTPException(400, "Invalid split_index. Must be >= 0.")

    subs = get_exam_submissions(eid, sid)
    source_sub = next((s for s in subs if s["question_id"].upper() == source_qid), None)
    if not source_sub:
        raise HTTPException(404, f"Source question '{source_qid}' not found for student '{sid}'")

    image_paths = source_sub.get("image_paths", [])
    text_to_split = (
        payload.current_text
        if payload.current_text is not None
        else (source_sub.get("transcribed_text") or "")
    )

    # Case A: split_index == 0 or text_to_split[:split_index].strip() == ""
    # The entire text moves to target_question_id, and source_question_id is deleted.
    if payload.split_index == 0 or not text_to_split[:payload.split_index].strip():
        target_text = text_to_split.strip()
        save_student_answer(eid, sid, target_qid, image_paths, target_text)
        if target_qid != source_qid:
            delete_student_submission(eid, sid, source_qid)
    else:
        # Case B: split_index > 0
        first_part = text_to_split[:payload.split_index].strip()
        second_part = text_to_split[payload.split_index:].strip()
        save_student_answer(eid, sid, source_qid, image_paths, first_part)
        save_student_answer(eid, sid, target_qid, image_paths, second_part)

    updated_subs = get_exam_submissions(eid, sid)
    for row in updated_subs:
        row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in updated_subs}
    ia_breakdown = calculate_ia_total(scores)

    return {
        "status": "success",
        "submissions": updated_subs,
        "ia_total": ia_breakdown["ia_total"],
        "ia_breakdown": ia_breakdown,
    }


@app.post("/api/submissions/evaluate-student")
def evaluate_student_endpoint(payload: EvaluateStudentRequest):
    """
    Grade one or all assigned question answers using the BERT rubric grader.
    """
    eid = payload.exam_id.strip().upper()
    sid = payload.student_id.strip().upper()
    rubrics = get_exam_rubrics(eid)
    rubric_map = {r["question_id"].upper(): r for r in rubrics}

    subs = get_exam_submissions(eid, sid)
    if payload.question_id:
        target_q = payload.question_id.strip().upper()
        subs_to_eval = [s for s in subs if s["question_id"].upper() == target_q]
    else:
        subs_to_eval = subs

    try:
        for s in subs_to_eval:
            q_id = s["question_id"].upper()
            ans_text = s.get("transcribed_text") or ""
            if q_id in rubric_map:
                r_info = rubric_map[q_id]
                eval_res = engine.evaluate_answer(
                    question_text=r_info["question_text"],
                    max_marks=r_info["max_marks"],
                    key_points=r_info["key_points"],
                    student_answer=ans_text,
                    reference_answer=r_info.get("reference_answer", ""),
                )
                update_evaluation(
                    exam_id=eid,
                    student_id=sid,
                    question_id=q_id,
                    matched_points=eval_res["matched_points"],
                    missing_points=eval_res["missing_points"],
                    ai_score=eval_res["ai_score"],
                    ai_feedback=eval_res["ai_feedback"],
                    similarity_score=eval_res["similarity_score"],
                    keyword_coverage=eval_res["keyword_coverage"],
                )
    finally:
        engine.unload_model()

    updated_subs = get_exam_submissions(eid, sid)
    for row in updated_subs:
        row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in updated_subs}
    ia_breakdown = calculate_ia_total(scores)

    return {
        "status": "success",
        "submissions": updated_subs,
        "ia_total": ia_breakdown["ia_total"],
        "ia_breakdown": ia_breakdown,
    }


@app.post("/api/submissions/assign-segment")
def assign_segment(payload: AssignSegmentRequest):
    try:
        res = assign_segment_to_question(
            exam_id=payload.exam_id,
            student_id=payload.student_id,
            question_id=payload.question_id,
            start_char=payload.start_char,
            end_char=payload.end_char,
            custom_text=payload.custom_text,
            remove_from_unassigned=payload.remove_from_unassigned,
        )
        for row in res["submissions"]:
            row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
        if res.get("booklet"):
            res["booklet"]["image_urls"] = [public_file_url(p) for p in res["booklet"].get("image_paths", [])]
        return res
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Assignment failed: {exc}")


@app.post("/api/submissions/split-at-cursor")
def split_at_cursor_endpoint(payload: SplitAtCursorRequest):
    try:
        res = split_at_cursor(
            exam_id=payload.exam_id,
            student_id=payload.student_id,
            source_question_id=payload.source_question_id,
            cursor_index=payload.cursor_index,
            target_question_id=payload.target_question_id,
        )
        for row in res["submissions"]:
            row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
        return res
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Split at cursor failed: {exc}")


@app.delete("/api/submissions/{exam_id}/{student_id}/{question_id}")
def delete_submission_endpoint(exam_id: str, student_id: str, question_id: str):
    try:
        res = delete_student_submission(exam_id, student_id, question_id)
        for row in res["submissions"]:
            row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
        return res
    except Exception as exc:
        raise HTTPException(500, f"Delete failed: {exc}")


@app.post("/api/submissions/merge-reassign")
def merge_reassign(payload: MergeReassignRequest):
    try:
        res = merge_and_reassign_submissions(
            exam_id=payload.exam_id,
            student_id=payload.student_id,
            source_question_ids=payload.source_question_ids,
            target_question_id=payload.target_question_id,
        )
        for row in res["submissions"]:
            row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
        return res
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Merge failed: {exc}")


@app.get("/api/sample-booklets")
def list_sample_booklets():
    samples = []
    if ROOT_TEST_DIR.exists():
        for p in ROOT_TEST_DIR.glob("*.pdf"):
            if "RMIPR" in p.name:
                continue
            samples.append({
                "filename": p.name,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                "description": "Student Answer Booklet (scanned multi-page PDF)",
            })
    return {"samples": samples}


def _run_grade_job(job_id: str, exam_id: str, student_id: str, page_paths: List[str]):
    job = jobs[job_id]
    try:
        with grade_lock:
            def progress(msg: str) -> None:
                job["progress"] = msg

            job["status"] = "running"
            result = _transcribe_and_grade(exam_id, student_id, page_paths, progress)
            job["transcript"] = result["transcript"]
            job["chunks"] = result["chunks"]
            job["results"] = result["submissions"]
            job["ia"] = result["ia"]
            job["needs_review"] = result["needs_review"]
            job["status"] = "done"
            job["progress"] = result["message"]
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["progress"] = f"Failed: {exc}"


@app.post("/api/exams/{exam_id}/grade")
async def grade_booklet(
    exam_id: str,
    background_tasks: BackgroundTasks,
    student_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    exam = get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    usn = student_id.strip().upper()
    if not usn:
        raise HTTPException(400, "Student USN is required")
    if not files:
        raise HTTPException(400, "Upload a booklet PDF or page images")

    page_paths = await _store_uploads(exam_id.strip().upper(), usn, files)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": "Queued",
        "exam_id": exam_id.upper(),
        "student_id": usn,
        "pages": len(page_paths),
        "transcript": "",
        "chunks": [],
        "results": [],
    }
    background_tasks.add_task(_run_grade_job, job_id, exam_id.upper(), usn, page_paths)
    return {"job_id": job_id, "pages": len(page_paths)}


@app.post("/api/exams/{exam_id}/grade-sample")
def grade_sample_booklet(
    exam_id: str,
    payload: SampleGradeIn,
    background_tasks: BackgroundTasks,
):
    exam = get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Exam not found")
    usn = payload.student_id.strip().upper()
    pdf_path = ROOT_TEST_DIR / payload.sample_name
    if not pdf_path.exists():
        raise HTTPException(404, f"Sample file '{payload.sample_name}' not found on server")

    raw_pdf = pdf_path.read_bytes()
    sample_dir = DATA_DIR / "booklets" / exam_id.upper() / usn / "sample"
    page_paths = convert_pdf_to_images(raw_pdf, str(sample_dir))
    if payload.max_pages and payload.max_pages > 0:
        page_paths = page_paths[:payload.max_pages]

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": "Queued",
        "exam_id": exam_id.upper(),
        "student_id": usn,
        "pages": len(page_paths),
        "transcript": "",
        "chunks": [],
        "results": [],
    }
    background_tasks.add_task(_run_grade_job, job_id, exam_id.upper(), usn, page_paths)
    return {"job_id": job_id, "pages": len(page_paths)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/api/verify")
def verify(payload: VerifyIn):
    verify_submission(
        payload.exam_id,
        payload.student_id,
        payload.question_id,
        payload.teacher_score,
        payload.feedback,
    )
    updated_points = update_rubric_weights_from_feedback(
        payload.exam_id,
        payload.student_id,
        payload.question_id,
        payload.teacher_score,
    )
    return {"ok": True, "key_points": updated_points}


@app.get("/api/exams/{exam_id}/ia/{student_id}")
def student_ia(exam_id: str, student_id: str):
    rows = get_exam_submissions(exam_id, student_id)
    scores = {r["question_id"]: float(r["teacher_score"] or 0.0) for r in rows}
    return {"scores": scores, **calculate_ia_total(scores), "answered": list(scores.keys())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
