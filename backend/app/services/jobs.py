"""Minimal background-job runner: a thread per job, progress stored in the Job table.

No Redis/Celery: for a single-server deployment a thread is enough and keeps setup
simple. Tests (ASYNC_JOBS=False) run jobs inline so results are deterministic.
"""
import threading
import traceback

from flask import current_app

from ..extensions import db
from ..models import Job


def create_job(kind: str, ref_id: int) -> Job:
    job = Job(kind=kind, ref_id=ref_id, status="queued", message="Queued")
    db.session.add(job)
    db.session.commit()
    return job


def update_job(job_id: int, **fields):
    job = db.session.get(Job, job_id)
    if job:
        for k, v in fields.items():
            setattr(job, k, v)
        db.session.commit()


def run_job(job: Job, func):
    """Run func(progress_cb) now (tests) or in a background thread."""
    app = current_app._get_current_object()
    job_id = job.id

    def target():
        with app.app_context():
            def progress(frac: float, msg: str = ""):
                update_job(job_id, progress=round(float(frac), 3), message=msg[:300])

            try:
                update_job(job_id, status="running", message="Running")
                func(progress)
                update_job(job_id, status="done", progress=1.0, message="Done")
            except Exception as exc:
                db.session.rollback()
                app.logger.error("job %s failed:\n%s", job_id, traceback.format_exc())
                update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}", message="Failed")
            finally:
                db.session.remove()

    if app.config.get("ASYNC_JOBS", True):
        threading.Thread(target=target, daemon=True, name=f"job-{job_id}").start()
    else:
        target()
