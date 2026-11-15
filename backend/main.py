import os
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
    create_exam,
    delete_exam_rubrics,
    get_all_exams,
    get_exam,
    get_exam_rubrics,
    get_exam_submissions,
    get_kpi_counts,
    get_verified_examples,
    init_db,
    save_question_rubric,
    save_student_answer,
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
    max_marks: int = 5
    key_points: Optional[List[Any]] = None


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
    max_marks: int
    key_points: List[Any]


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
        save_question_rubric(exam_id, qid, q.question_text.strip(), int(q.max_marks), points)
        saved.append({"question_id": qid, "question_text": q.question_text, "max_marks": q.max_marks, "key_points": points})
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
    save_question_rubric(exam_id, question_id, payload.question_text, payload.max_marks, payload.key_points)
    return {"ok": True}


@app.get("/api/exams/{exam_id}/kpis")
def exam_kpis(exam_id: str):
    return get_kpi_counts(exam_id)


@app.get("/api/exams/{exam_id}/submissions")
def exam_submissions(exam_id: str, student_id: Optional[str] = None):
    rows = get_exam_submissions(exam_id, student_id)
    for row in rows:
        row["image_urls"] = [public_file_url(p) for p in row.get("image_paths", [])]
    students = sorted({r["student_id"] for r in rows})
    return {"students": students, "submissions": rows}


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
            rubrics = get_exam_rubrics(exam_id)
            if not rubrics:
                raise RuntimeError("No questions configured for this exam.")

            def progress(msg: str):
                job["progress"] = msg

            job["status"] = "running"
            job["progress"] = "Starting booklet digitization with Qwen2-VL"
            full_md = engine.transcribe_images(page_paths, progress_cb=progress)
            job["transcript"] = full_md

            job["progress"] = "Segmenting answers with hybrid boundary detection"
            chunks = chunk_student_booklet(full_md, rubrics=rubrics)
            if not chunks:
                raise RuntimeError("Could not segment the booklet into questions.")

            relative_paths = []
            for p in page_paths:
                try:
                    rel = Path(p).resolve().relative_to(DATA_DIR.resolve()).as_posix()
                    relative_paths.append(rel)
                except Exception:
                    relative_paths.append(p)

            rubric_map = {r["question_id"].upper(): r for r in rubrics}
            results = []
            for q_id, ans_text in chunks.items():
                job["progress"] = f"Grading {q_id}"
                save_student_answer(exam_id, student_id, q_id, relative_paths, ans_text)
                if q_id in rubric_map:
                    r_info = rubric_map[q_id]
                    past = get_verified_examples(exam_id, q_id, limit=2)
                    eval_res = engine.evaluate_answer(
                        question_text=r_info["question_text"],
                        max_marks=r_info["max_marks"],
                        key_points=r_info["key_points"],
                        student_answer=ans_text,
                        verified_examples=past,
                    )
                    update_evaluation(
                        exam_id=exam_id,
                        student_id=student_id,
                        question_id=q_id,
                        matched_points=eval_res["matched_points"],
                        missing_points=eval_res["missing_points"],
                        ai_score=eval_res["ai_score"],
                        ai_feedback=eval_res["ai_feedback"],
                    )
                    results.append({"question_id": q_id, **eval_res})
                else:
                    results.append({"question_id": q_id, "ai_score": None, "ai_feedback": "No matching rubric."})

            scores = {r["question_id"]: float(r.get("ai_score") or 0) for r in results if r.get("ai_score") is not None}
            job["ia"] = calculate_ia_total(scores)
            job["chunks"] = list(chunks.keys())
            job["results"] = results
            job["status"] = "done"
            job["progress"] = "Evaluation complete"
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

    student_dir = DATA_DIR / "booklets" / exam_id.upper() / usn
    student_dir.mkdir(parents=True, exist_ok=True)
    page_paths: List[str] = []
    img_idx = 0
    for upload in files:
        raw = await upload.read()
        name = (upload.filename or "page.bin").lower()
        if name.endswith(".pdf"):
            page_paths.extend(convert_pdf_to_images(raw, str(student_dir)))
        else:
            img_idx += 1
            ext = Path(name).suffix or ".png"
            dest = student_dir / f"page_{img_idx}{ext}"
            dest.write_bytes(raw)
            page_paths.append(str(dest))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": "Queued",
        "exam_id": exam_id.upper(),
        "student_id": usn,
        "pages": len(page_paths),
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

    student_dir = DATA_DIR / "booklets" / exam_id.upper() / usn
    student_dir.mkdir(parents=True, exist_ok=True)

    raw_pdf = pdf_path.read_bytes()
    page_paths = convert_pdf_to_images(raw_pdf, str(student_dir))
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
