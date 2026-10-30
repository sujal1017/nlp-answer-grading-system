"""Database models.

Design rule: original OCR text, corrected OCR text, AI scores and human scores are
stored in separate columns so nothing is ever overwritten silently.
"""
from datetime import datetime, timezone

from .extensions import db


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TimestampMixin:
    created_at = db.Column(db.DateTime, default=_now, nullable=False)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now, nullable=False)


class Setting(db.Model):
    key = db.Column(db.String(80), primary_key=True)
    value = db.Column(db.JSON, nullable=False)


class Exam(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(200), default="")
    code = db.Column(db.String(60), default="")
    notes = db.Column(db.Text, default="")
    instructions = db.Column(db.Text, default="")
    # e.g. [["Q1","Q2"]] = "answer any one of Q1 or Q2": the best-scoring main question counts.
    choice_groups = db.Column(db.JSON, default=list)

    questions = db.relationship(
        "Question", backref="exam", cascade="all, delete-orphan", order_by="Question.position"
    )
    documents = db.relationship("Document", backref="exam", cascade="all, delete-orphan")
    submissions = db.relationship("Submission", backref="exam", cascade="all, delete-orphan")


class Document(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exam.id"), nullable=False)
    kind = db.Column(db.String(30), nullable=False)  # question_paper | answer_sheet
    original_filename = db.Column(db.String(300), nullable=False)
    stored_name = db.Column(db.String(100), nullable=False)  # server-generated, never user-supplied
    sha256 = db.Column(db.String(64), nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False)
    page_count = db.Column(db.Integer, nullable=False)
    page_analysis = db.Column(db.JSON, default=list)  # per-page type info


class Question(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exam.id"), nullable=False)
    position = db.Column(db.Integer, default=0, nullable=False)
    label = db.Column(db.String(40), nullable=False)  # e.g. "Q1A", "Q3"
    text = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text, default="")  # parent question text / instructions
    max_marks = db.Column(db.Float, nullable=True)  # null => unknown, must be confirmed
    marks_confirmed = db.Column(db.Boolean, default=False, nullable=False)
    qtype = db.Column(db.String(30), default="unknown", nullable=False)
    source_page = db.Column(db.Integer, nullable=True)
    source = db.Column(db.String(20), default="manual", nullable=False)  # extracted | manual
    needs_review = db.Column(db.Boolean, default=False, nullable=False)
    review_reasons = db.Column(db.JSON, default=list)

    reference_answer = db.Column(db.Text, default="")
    reference_source = db.Column(db.String(20), default="none")  # none | human | ai_draft
    reference_approved = db.Column(db.Boolean, default=False, nullable=False)

    criteria = db.relationship(
        "Criterion", backref="question", cascade="all, delete-orphan", order_by="Criterion.position"
    )

    __table_args__ = (db.UniqueConstraint("exam_id", "label", name="uq_question_label_per_exam"),)


class Criterion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey("question.id"), nullable=False)
    position = db.Column(db.Integer, default=0, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, default="")
    max_marks = db.Column(db.Float, nullable=False)
    # list of {"id": str, "text": str, "keywords": [str], "synonyms": [str]}
    concepts = db.Column(db.JSON, default=list)

    __table_args__ = (db.CheckConstraint("max_marks >= 0", name="ck_criterion_marks_nonneg"),)


class Submission(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey("exam.id"), nullable=False)
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"), nullable=False)
    student_identifier = db.Column(db.String(100), default="")
    # uploaded | ocr_running | ocr_done | ocr_failed | mapped | graded | reviewed
    status = db.Column(db.String(30), default="uploaded", nullable=False)
    mapping_notes = db.Column(db.JSON, default=list)
    unmapped_text = db.Column(db.Text, default="")

    document = db.relationship("Document")
    pages = db.relationship(
        "PageText", backref="submission", cascade="all, delete-orphan", order_by="PageText.page_number"
    )
    answers = db.relationship("Answer", backref="submission", cascade="all, delete-orphan")


class PageText(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submission.id"), nullable=False)
    page_number = db.Column(db.Integer, nullable=False)  # 1-based
    page_kind = db.Column(db.String(30), default="unknown")  # text | scanned | mixed | empty
    method = db.Column(db.String(40), default="")  # direct_text | ocr engine name | none
    engine_info = db.Column(db.String(200), default="")
    text_original = db.Column(db.Text, default="")
    text_corrected = db.Column(db.Text, nullable=True)  # null => reviewer made no correction
    confidence = db.Column(db.Float, nullable=True)  # only if genuinely provided by engine
    warnings = db.Column(db.JSON, default=list)
    error = db.Column(db.Text, nullable=True)
    reviewed = db.Column(db.Boolean, default=False, nullable=False)

    __table_args__ = (db.UniqueConstraint("submission_id", "page_number", name="uq_page_per_submission"),)

    @property
    def effective_text(self) -> str:
        return self.text_corrected if self.text_corrected is not None else (self.text_original or "")


class Answer(TimestampMixin, db.Model):
    """The student's answer to one question, after mapping."""

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submission.id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("question.id"), nullable=False)
    text_original = db.Column(db.Text, default="")
    text_corrected = db.Column(db.Text, nullable=True)
    pages = db.Column(db.JSON, default=list)
    # answered | unanswered | unreadable | ambiguous
    status = db.Column(db.String(20), default="unanswered", nullable=False)
    mapping_confirmed = db.Column(db.Boolean, default=False, nullable=False)
    notes = db.Column(db.JSON, default=list)

    question = db.relationship("Question")
    evaluation = db.relationship(
        "Evaluation", backref="answer", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (db.UniqueConstraint("submission_id", "question_id", name="uq_answer_per_question"),)

    @property
    def effective_text(self) -> str:
        return self.text_corrected if self.text_corrected is not None else (self.text_original or "")


class Evaluation(TimestampMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    answer_id = db.Column(db.Integer, db.ForeignKey("answer.id"), nullable=False, unique=True)
    # pending | ai_graded | needs_review | manual_required | reviewed
    status = db.Column(db.String(20), default="pending", nullable=False)
    # --- AI output (never overwritten by human edits) ---
    ai_total = db.Column(db.Float, nullable=True)
    ai_result = db.Column(db.JSON, nullable=True)  # full validated EvaluationResult
    ai_error = db.Column(db.Text, nullable=True)
    ai_flags = db.Column(db.JSON, default=list)
    judge_name = db.Column(db.String(60), default="")
    embedder_name = db.Column(db.String(60), default="")
    attempts = db.Column(db.Integer, default=0)
    graded_at = db.Column(db.DateTime, nullable=True)
    # --- Human review ---
    human_criterion_scores = db.Column(db.JSON, nullable=True)  # {criterion_id: marks}
    human_total = db.Column(db.Float, nullable=True)
    reviewer_comment = db.Column(db.Text, default="")
    reviewed_at = db.Column(db.DateTime, nullable=True)
    approved = db.Column(db.Boolean, default=False, nullable=False)
    # --- Final (set only when a human approves) ---
    final_total = db.Column(db.Float, nullable=True)


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(40), nullable=False)
    ref_id = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(20), default="queued", nullable=False)  # queued|running|done|failed
    progress = db.Column(db.Float, default=0.0)
    message = db.Column(db.String(300), default="")
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=_now, nullable=False)
    updated_at = db.Column(db.DateTime, default=_now, onupdate=_now, nullable=False)
