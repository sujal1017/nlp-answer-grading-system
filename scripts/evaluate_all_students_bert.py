#!/usr/bin/env python
"""Grade a SQLite exam database with the project's BERT rubric grader."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from bert_scorer import MODEL_ID, _load_model, evaluate_with_bert  # noqa: E402
from chunker import calculate_ia_total, chunk_student_booklet  # noqa: E402


LEDGER_QUESTIONS = ("Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B")
QUESTION_MARKER_RE = re.compile(
    r"(?im)^\s*(?:(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)\d+"
    r"(?:\s*(?:[A-D]|\(\s*[A-D]\s*\))?)|\d+\s*[\[(]\s*[A-D]\s*[\])])"
    r"(?=\s|[):.\]-]|$)"
)

LEDGER_FIELDS = [
    "Student USN",
    *LEDGER_QUESTIONS,
    "Q1 Total",
    "Q2 Total",
    "Best (Q1/Q2)",
    "Q3 Total",
    "Final IA Total",
    "Max Marks",
    "Verified Sum",
    "Answered",
    "Status",
    "Last Updated",
]

LOG_FIELDS = [
    "Student USN",
    "Submission ID",
    "Question ID",
    "Student Answer",
    "Reference Answer",
    "Maximum Marks",
    "BERT Similarity Score",
    "Keyword Coverage",
    "Awarded Marks",
    "Evaluation Status",
    "Error/Skip Reason",
]


class EvaluationError(RuntimeError):
    """Raised when the database cannot be graded safely."""


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_columns(connection: sqlite3.Connection, table: str) -> Dict[str, str]:
    rows = connection.execute(
        "PRAGMA table_info(%s)" % quote_identifier(table)
    ).fetchall()
    return {str(row[1]).casefold(): str(row[1]) for row in rows}


def find_table(
    schema: Dict[str, Dict[str, str]],
    required_columns: Iterable[str],
    preferred_names: Sequence[str],
) -> Tuple[str, Dict[str, str]]:
    required = {column.casefold() for column in required_columns}
    for preferred in preferred_names:
        columns = schema.get(preferred.casefold())
        if columns and required.issubset(columns):
            return preferred, columns
    for table_name, columns in schema.items():
        if required.issubset(columns):
            return table_name, columns
    raise EvaluationError(
        "Could not identify a table with columns: " + ", ".join(sorted(required))
    )


def row_value(
    row: sqlite3.Row,
    columns: Dict[str, str],
    logical_name: str,
    default: Any = None,
) -> Any:
    actual_name = columns.get(logical_name.casefold())
    return row[actual_name] if actual_name is not None else default


def read_table(
    connection: sqlite3.Connection, table: str
) -> List[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM %s" % quote_identifier(table)
    ).fetchall()


def connect_read_only(database_path: Path) -> sqlite3.Connection:
    uri = database_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def select_database(requested_path: Optional[Path]) -> Path:
    if requested_path is not None:
        candidate = requested_path.expanduser().resolve()
        if not candidate.is_file():
            raise EvaluationError("Database file does not exist: %s" % candidate)
        return candidate

    environment_path = os.environ.get("GRADING_DB")
    if environment_path:
        candidate = Path(environment_path).expanduser().resolve()
        if not candidate.is_file():
            raise EvaluationError(
                "GRADING_DB points to a missing file: %s" % candidate
            )
        return candidate

    candidates = [
        Path.home() / "Downloads" / "grading_system.db",
        BACKEND_DIR / "grading_system.db",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise EvaluationError(
        "No database found. Pass --db PATH or set GRADING_DB. "
        "Default locations checked: %s"
        % ", ".join(str(path) for path in candidates)
    )


def normalize_question_id(raw_question_id: Any, rubric_ids: Sequence[str]) -> Optional[str]:
    raw = str(raw_question_id or "").strip().upper()
    known = set(rubric_ids)
    if raw in known:
        return raw

    subquestion = re.fullmatch(
        r"(?:QUESTION|Q|ANSWER|ANS)?\.?\s*(\d+)\s*"
        r"(?:\(\s*([A-D])\s*\)|([A-D]))",
        raw,
    )
    if subquestion:
        candidate = "Q%s%s" % (
            subquestion.group(1),
            subquestion.group(2) or subquestion.group(3),
        )
        return candidate if candidate in known else None

    parent_question = re.fullmatch(
        r"(?:QUESTION|Q|ANSWER|ANS)?\.?\s*(\d+)", raw
    )
    if parent_question:
        prefix = "Q%s" % parent_question.group(1)
        matches = [question_id for question_id in rubric_ids if question_id.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
    return None


def sort_question_ids(question_ids: Iterable[str]) -> List[str]:
    def key(question_id: str) -> Tuple[int, str]:
        match = re.fullmatch(r"Q(\d+)([A-Z])", question_id)
        if match:
            return int(match.group(1)), match.group(2)
        return (10**9, question_id)

    return sorted(question_ids, key=key)


def rubric_reference(rubric: Dict[str, Any]) -> str:
    explicit = str(rubric.get("reference_answer") or "").strip()
    if explicit:
        return explicit
    points = rubric.get("key_points") or []
    point_texts = [
        str(point.get("point", "")).strip()
        if isinstance(point, dict)
        else str(point).strip()
        for point in points
    ]
    return ". ".join(text for text in point_texts if text)


def parse_key_points(value: Any, question_id: str) -> List[Any]:
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            "Invalid rubric key_points for %s: %s" % (question_id, exc)
        ) from exc
    if decoded is None:
        return []
    if not isinstance(decoded, list):
        raise EvaluationError(
            "Rubric key_points for %s must be a JSON list." % question_id
        )
    return decoded


def excel_safe_text(value: Any) -> str:
    """Prevent answer text from being interpreted as a spreadsheet formula."""
    text = "" if value is None else str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8-sig",
            suffix=".tmp",
            prefix=path.stem + "_",
            dir=str(path.parent),
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary_path), str(path))
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def evaluate_database(
    database_path: Path,
    exam_id_arg: Optional[str],
    ledger_path: Path,
    log_path: Path,
) -> int:
    connection = connect_read_only(database_path)
    log_rows: List[Dict[str, Any]] = []
    try:
        table_names = {
            str(row[0]).casefold(): str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        schema = {
            table_name: table_columns(connection, actual_name)
            for table_name, actual_name in table_names.items()
        }
        submissions_table, submissions_columns = find_table(
            schema,
            ("student_id", "question_id", "transcribed_text"),
            ("submissions",),
        )
        rubrics_table, rubrics_columns = find_table(
            schema,
            ("question_id", "question_text", "max_marks", "key_points"),
            ("rubrics",),
        )

        exam_ids = set()
        if "exam_id" in submissions_columns:
            exam_ids.update(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL"
                    % (
                        quote_identifier(submissions_columns["exam_id"]),
                        quote_identifier(submissions_table),
                        quote_identifier(submissions_columns["exam_id"]),
                    )
                )
            )
        rubric_exam_ids = set()
        if "exam_id" in rubrics_columns:
            rubric_exam_ids.update(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL"
                    % (
                        quote_identifier(rubrics_columns["exam_id"]),
                        quote_identifier(rubrics_table),
                        quote_identifier(rubrics_columns["exam_id"]),
                    )
                )
            )
        exam_table = table_names.get("exams")
        exam_columns = schema.get(exam_table.casefold(), {}) if exam_table else {}
        canonical_exam_ids = set()
        if exam_table and "exam_id" in exam_columns:
            canonical_exam_ids.update(
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT %s FROM %s WHERE %s IS NOT NULL"
                    % (
                        quote_identifier(exam_columns["exam_id"]),
                        quote_identifier(exam_table),
                        quote_identifier(exam_columns["exam_id"]),
                    )
                )
            )
        if not exam_ids and not rubric_exam_ids:
            exam_ids.update(canonical_exam_ids)
        if not exam_ids and rubric_exam_ids:
            exam_ids.update(rubric_exam_ids)

        known_exam_ids = exam_ids | rubric_exam_ids | canonical_exam_ids
        if exam_id_arg:
            exam_id = exam_id_arg
            if known_exam_ids and exam_id not in known_exam_ids:
                raise EvaluationError(
                    "Exam ID %s is not present in the selected database." % exam_id
                )
        elif len(canonical_exam_ids) == 1:
            # The exams table is authoritative. Some databases also contain
            # isolated test submissions for exams with no exam/rubric record.
            exam_id = next(iter(canonical_exam_ids))
        elif len(canonical_exam_ids) > 1:
            raise EvaluationError(
                "Database contains multiple registered exams (%s); pass --exam-id."
                % ", ".join(sorted(canonical_exam_ids))
            )
        elif len(known_exam_ids) == 1:
            exam_id = next(iter(known_exam_ids))
        elif not known_exam_ids:
            exam_id = ""
        else:
            raise EvaluationError(
                "Database contains multiple exams (%s); pass --exam-id."
                % ", ".join(sorted(known_exam_ids))
            )

        rubric_rows = read_table(connection, rubrics_table)
        rubrics: Dict[str, Dict[str, Any]] = {}
        segmenter_rubrics: List[Dict[str, Any]] = []
        for row in rubric_rows:
            row_exam_id = row_value(row, rubrics_columns, "exam_id")
            if exam_id and row_exam_id is not None and str(row_exam_id) != exam_id:
                continue
            question_id = str(row_value(row, rubrics_columns, "question_id") or "").strip().upper()
            if not question_id:
                raise EvaluationError("A rubric row has an empty question_id.")
            if question_id in rubrics:
                raise EvaluationError("Duplicate rubric found for %s." % question_id)
            key_points = parse_key_points(
                row_value(row, rubrics_columns, "key_points"), question_id
            )
            max_marks = float(row_value(row, rubrics_columns, "max_marks"))
            if max_marks < 0:
                raise EvaluationError("Negative maximum marks for %s." % question_id)
            rubric = {
                "question_id": question_id,
                "question_text": str(
                    row_value(row, rubrics_columns, "question_text") or ""
                ),
                "max_marks": max_marks,
                "key_points": key_points,
                "reference_answer": (
                    str(row_value(row, rubrics_columns, "reference_answer") or "").strip()
                    if "reference_answer" in rubrics_columns
                    else ""
                ),
            }
            rubrics[question_id] = rubric
            segmenter_rubrics.append(rubric)

        question_ids = sort_question_ids(rubrics)
        if not question_ids:
            raise EvaluationError("No rubrics were found for the selected exam.")
        if set(question_ids) != set(LEDGER_QUESTIONS):
            raise EvaluationError(
                "This ledger's total formula expects %s; database rubrics contain %s."
                % (", ".join(LEDGER_QUESTIONS), ", ".join(question_ids))
            )
        question_ids = list(LEDGER_QUESTIONS)

        all_submission_rows = read_table(connection, submissions_table)
        submission_rows = [
            row
            for row in all_submission_rows
            if not exam_id
            or "exam_id" not in submissions_columns
            or str(row_value(row, submissions_columns, "exam_id") or "") == exam_id
        ]
        excluded_submission_rows = (
            [
                row
                for row in all_submission_rows
                if str(row_value(row, submissions_columns, "exam_id") or "") != exam_id
            ]
            if exam_id and "exam_id" in submissions_columns
            else []
        )
        submissions_by_student: Dict[str, List[sqlite3.Row]] = defaultdict(list)
        for row in submission_rows:
            student_id = str(row_value(row, submissions_columns, "student_id") or "").strip()
            if not student_id:
                raise EvaluationError(
                    "Submission %s has no student ID; refusing to omit it."
                    % row_value(row, submissions_columns, "submission_id", "unknown")
                )
            submissions_by_student[student_id].append(row)

        booklets_table = None
        booklets_columns: Dict[str, str] = {}
        booklets_by_student: Dict[str, sqlite3.Row] = {}
        other_exam_booklet_ids = set()
        try:
            booklets_table, booklets_columns = find_table(
                schema,
                ("student_id", "full_transcribed_text"),
                ("booklets",),
            )
        except EvaluationError:
            pass
        if booklets_table:
            for row in read_table(connection, booklets_table):
                row_exam_id = row_value(row, booklets_columns, "exam_id")
                student_id = str(row_value(row, booklets_columns, "student_id") or "").strip()
                if not student_id:
                    raise EvaluationError(
                        "A booklet record has no student ID; refusing to omit it."
                    )
                if exam_id and row_exam_id is not None and str(row_exam_id) != exam_id:
                    other_exam_booklet_ids.add(student_id)
                    continue
                if student_id in booklets_by_student:
                    raise EvaluationError(
                        "Multiple full booklet records exist for student %s in exam %s."
                        % (student_id, exam_id)
                    )
                booklets_by_student[student_id] = row

        student_results_table = table_names.get("student_results")
        student_results_columns = (
            schema.get(student_results_table.casefold(), {})
            if student_results_table
            else {}
        )
        result_student_ids = set()
        other_exam_result_ids = set()
        if student_results_table and "student_id" in student_results_columns:
            for row in read_table(connection, student_results_table):
                row_exam_id = row_value(row, student_results_columns, "exam_id")
                student_id = str(row_value(row, student_results_columns, "student_id") or "").strip()
                if student_id:
                    if exam_id and row_exam_id is not None and str(row_exam_id) != exam_id:
                        other_exam_result_ids.add(student_id)
                    else:
                        result_student_ids.add(student_id)
                else:
                    raise EvaluationError(
                        "A student_results record has no student ID; refusing to omit it."
                    )

        selected_exam_student_ids = (
            set(submissions_by_student) | set(booklets_by_student) | result_student_ids
        )
        other_exam_student_ids = {
            str(row_value(row, submissions_columns, "student_id") or "").strip()
            for row in excluded_submission_rows
        }
        other_exam_student_ids.discard("")
        other_exam_student_ids.update(other_exam_booklet_ids)
        other_exam_student_ids.update(other_exam_result_ids)
        ungradable_exam_only_students = other_exam_student_ids - selected_exam_student_ids
        student_ids = sorted(selected_exam_student_ids | other_exam_student_ids)
        if not student_ids:
            raise EvaluationError("No student IDs were found in the selected exam.")

        # Load the project model before writing results. A load failure is fatal;
        # there is deliberately no substitute model or scoring fallback.
        print("Loading BERT model: %s" % MODEL_ID, flush=True)
        _load_model()
        print("BERT model loaded; starting student-answer evaluation.", flush=True)

        marks: Dict[str, Dict[str, Optional[float]]] = {
            student_id: {question_id: 0.0 for question_id in question_ids}
            for student_id in student_ids
        }
        identified_questions: Dict[str, set] = {
            student_id: set() for student_id in student_ids
        }
        answered_questions: Dict[str, set] = {
            student_id: set() for student_id in student_ids
        }
        review_students = set()
        unresolved_students = set()
        uncertain_assignment_students = set()
        evaluated_answers = 0
        failed_answers = 0
        processed_answer_entries = 0
        detected_answers = len(all_submission_rows)

        def append_log(
            student_id: str,
            submission_id: Any,
            question_id: str,
            answer: Any,
            reference: str = "",
            max_marks: Any = "",
            similarity: Any = "",
            keyword_coverage: Any = "",
            awarded: Any = "",
            status: str = "",
            reason: str = "",
        ) -> None:
            log_rows.append(
                {
                    "Student USN": excel_safe_text(student_id),
                    "Submission ID": excel_safe_text(submission_id),
                    "Question ID": excel_safe_text(question_id),
                    "Student Answer": excel_safe_text(answer),
                    "Reference Answer": excel_safe_text(reference),
                    "Maximum Marks": max_marks,
                    "BERT Similarity Score": similarity,
                    "Keyword Coverage": keyword_coverage,
                    "Awarded Marks": awarded,
                    "Evaluation Status": status,
                    "Error/Skip Reason": excel_safe_text(reason),
                }
            )

        for row in excluded_submission_rows:
            other_exam_id = row_value(row, submissions_columns, "exam_id", "")
            append_log(
                student_id=str(
                    row_value(row, submissions_columns, "student_id") or "UNKNOWN"
                ).strip(),
                submission_id=row_value(row, submissions_columns, "submission_id", ""),
                question_id=str(
                    row_value(row, submissions_columns, "question_id") or "UNMAPPED"
                ),
                answer=row_value(row, submissions_columns, "transcribed_text", ""),
                status="SKIPPED_OTHER_EXAM",
                reason=(
                    "Submission belongs to exam %s, not the selected exam %s; "
                    "that exam has no matching rubric in this run."
                    % (other_exam_id, exam_id)
                ),
            )

        for student_id in ungradable_exam_only_students:
            review_students.add(student_id)
            unresolved_students.add(student_id)
            for question_id in question_ids:
                marks[student_id][question_id] = None
                identified_questions[student_id].add(question_id)
                rubric = rubrics[question_id]
                append_log(
                    student_id=student_id,
                    submission_id="",
                    question_id=question_id,
                    answer="",
                    reference=rubric_reference(rubric),
                    max_marks=rubric["max_marks"],
                    status="SKIPPED_OTHER_EXAM",
                    reason=(
                        "No answer is assigned to the selected exam; this student "
                        "has records only for another exam without its own rubric."
                    ),
                )

        for student_id in (
            (other_exam_booklet_ids | other_exam_result_ids) - {
                str(row_value(row, submissions_columns, "student_id") or "").strip()
                for row in excluded_submission_rows
            }
        ):
            if student_id in ungradable_exam_only_students:
                append_log(
                    student_id=student_id,
                    submission_id="",
                    question_id="UNMAPPED",
                    answer="",
                    status="SKIPPED_OTHER_EXAM",
                    reason=(
                        "The student record belongs to a different exam without a "
                        "matching rubric in this run."
                    ),
                )

        def evaluate_one(
            student_id: str,
            submission_id: Any,
            question_id: str,
            answer: Any,
            source_status: str,
        ) -> None:
            nonlocal evaluated_answers, failed_answers, processed_answer_entries
            rubric = rubrics[question_id]
            answer_text = "" if answer is None else str(answer)
            processed_answer_entries += 1
            try:
                result = evaluate_with_bert(
                    question_text=rubric["question_text"],
                    max_marks=rubric["max_marks"],
                    key_points=rubric["key_points"],
                    student_answer=answer_text,
                    reference_answer=rubric_reference(rubric),
                )
                score = float(result["ai_score"])
                if not 0 <= score <= rubric["max_marks"]:
                    raise ValueError(
                        "BERT scorer returned %.4g outside 0..%.4g."
                        % (score, rubric["max_marks"])
                    )
                marks[student_id][question_id] = score
                identified_questions[student_id].add(question_id)
                evaluated_answers += 1
                if answer_text.strip():
                    answered_questions[student_id].add(question_id)
                if processed_answer_entries % 20 == 0:
                    print(
                        "Processed %d answer sections (%d evaluated, %d failed)."
                        % (processed_answer_entries, evaluated_answers, failed_answers),
                        flush=True,
                    )
                status = (
                    "EVALUATED_EMPTY"
                    if not answer_text.strip()
                    else source_status
                )
                append_log(
                    student_id=student_id,
                    submission_id=submission_id,
                    question_id=question_id,
                    answer=answer_text,
                    reference=rubric_reference(rubric),
                    max_marks=rubric["max_marks"],
                    similarity=result.get("similarity_score", 0.0),
                    keyword_coverage=result.get("keyword_coverage", 0.0),
                    awarded=score,
                    status=status,
                    reason=(
                        "The stored answer was blank; the existing BERT grader returned 0."
                        if status == "EVALUATED_EMPTY"
                        else ""
                    ),
                )
            except Exception as exc:
                marks[student_id][question_id] = None
                identified_questions[student_id].add(question_id)
                review_students.add(student_id)
                unresolved_students.add(student_id)
                failed_answers += 1
                append_log(
                    student_id=student_id,
                    submission_id=submission_id,
                    question_id=question_id,
                    answer=answer_text,
                    reference=rubric_reference(rubric),
                    max_marks=rubric["max_marks"],
                    status="FAILED",
                    reason="%s: %s" % (type(exc).__name__, exc),
                )

        # The submission table already stores question-level sections and is
        # authoritative when present. Do not re-segment those same booklets:
        # doing so could duplicate or contradict the saved question assignments.
        for student_id in student_ids:
            groups: Dict[str, List[sqlite3.Row]] = defaultdict(list)
            for row in submissions_by_student.get(student_id, []):
                raw_qid = row_value(row, submissions_columns, "question_id")
                question_id = normalize_question_id(raw_qid, question_ids)
                if question_id is None:
                    review_students.add(student_id)
                    unresolved_students.add(student_id)
                    uncertain_assignment_students.add(student_id)
                    raw_answer = row_value(row, submissions_columns, "transcribed_text", "")
                    append_log(
                        student_id=student_id,
                        submission_id=row_value(row, submissions_columns, "submission_id", ""),
                        question_id=str(raw_qid or "UNMAPPED"),
                        answer=raw_answer,
                        status="SKIPPED_UNMAPPED",
                        reason="The stored question ID could not be matched to a rubric.",
                    )
                    continue
                groups[question_id].append(row)

            for question_id, rows in groups.items():
                if len(rows) > 1:
                    identified_questions[student_id].add(question_id)
                    marks[student_id][question_id] = None
                    review_students.add(student_id)
                    unresolved_students.add(student_id)
                    reason = (
                        "%d submissions exist for this student/question; no version "
                        "was chosen because the database provides no reliable revision order."
                        % len(rows)
                    )
                    for row in rows:
                        append_log(
                            student_id=student_id,
                            submission_id=row_value(
                                row, submissions_columns, "submission_id", ""
                            ),
                            question_id=question_id,
                            answer=row_value(
                                row, submissions_columns, "transcribed_text", ""
                            ),
                            reference=rubric_reference(rubrics[question_id]),
                            max_marks=rubrics[question_id]["max_marks"],
                            status="SKIPPED_AMBIGUOUS_DUPLICATE",
                            reason=reason,
                        )
                    continue

                row = rows[0]
                evaluate_one(
                    student_id=student_id,
                    submission_id=row_value(
                        row, submissions_columns, "submission_id", ""
                    ),
                    question_id=question_id,
                    answer=row_value(
                        row, submissions_columns, "transcribed_text", ""
                    ),
                    source_status="EVALUATED_SUBMISSION",
                )

        # A booklet-only student may still have answers. Use the project's
        # existing question segmenter, but require a visible question marker so
        # its fuzzy/default routing cannot assign an arbitrary first question.
        booklets_without_submissions = [
            student_id
            for student_id in student_ids
            if not submissions_by_student.get(student_id)
            and student_id in booklets_by_student
        ]
        for index, student_id in enumerate(booklets_without_submissions, start=1):
            booklet = booklets_by_student[student_id]
            transcript = str(
                row_value(booklet, booklets_columns, "full_transcribed_text", "") or ""
            )
            booklet_id = row_value(booklet, booklets_columns, "id", "")
            submission_id = "BOOKLET:%s" % booklet_id if booklet_id != "" else "BOOKLET"
            if not transcript.strip():
                continue
            if not QUESTION_MARKER_RE.search(transcript):
                review_students.add(student_id)
                unresolved_students.add(student_id)
                uncertain_assignment_students.add(student_id)
                append_log(
                    student_id=student_id,
                    submission_id=submission_id,
                    question_id="UNMAPPED",
                    answer=transcript,
                    status="SKIPPED_UNCERTAIN_SEGMENTATION",
                    reason=(
                        "The booklet has no recognizable question marker; the project "
                        "segmenter would have to guess a question assignment."
                    ),
                )
                print(
                    "Review needed: %s has a booklet transcript but no question marker."
                    % student_id,
                    flush=True,
                )
                continue
            try:
                sections = chunk_student_booklet(transcript, segmenter_rubrics)
            except Exception as exc:
                review_students.add(student_id)
                unresolved_students.add(student_id)
                uncertain_assignment_students.add(student_id)
                append_log(
                    student_id=student_id,
                    submission_id=submission_id,
                    question_id="UNMAPPED",
                    answer=transcript,
                    status="FAILED_SEGMENTATION",
                    reason="%s: %s" % (type(exc).__name__, exc),
                )
                continue

            detected_answers += len(sections)
            if not sections:
                review_students.add(student_id)
                unresolved_students.add(student_id)
                uncertain_assignment_students.add(student_id)
                append_log(
                    student_id=student_id,
                    submission_id=submission_id,
                    question_id="UNMAPPED",
                    answer=transcript,
                    status="SKIPPED_UNMAPPED",
                    reason="Question markers were present, but no answer section was parsed.",
                )
            for raw_qid, answer in sections.items():
                question_id = normalize_question_id(raw_qid, question_ids)
                if question_id is None:
                    review_students.add(student_id)
                    unresolved_students.add(student_id)
                    uncertain_assignment_students.add(student_id)
                    append_log(
                        student_id=student_id,
                        submission_id=submission_id,
                        question_id=str(raw_qid),
                        answer=answer,
                        status="SKIPPED_UNMAPPED",
                        reason="The segmented question ID has no matching exam rubric.",
                    )
                    continue
                evaluate_one(
                    student_id=student_id,
                    submission_id=submission_id,
                    question_id=question_id,
                    answer=answer,
                    source_status="EVALUATED_BOOKLET_SEGMENT",
                )
            if index % 10 == 0:
                print(
                    "Processed %d/%d booklet-only students."
                    % (index, len(booklets_without_submissions)),
                    flush=True,
                )

        # Include every expected question in the detailed log. Unattempted
        # questions are zero in this exam's ledger; failed/ambiguous evaluations
        # remain blank and are not converted into arbitrary marks.
        for student_id in student_ids:
            for question_id in question_ids:
                if question_id in identified_questions[student_id]:
                    continue
                rubric = rubrics[question_id]
                if student_id in uncertain_assignment_students:
                    marks[student_id][question_id] = None
                    identified_questions[student_id].add(question_id)
                    append_log(
                        student_id=student_id,
                        submission_id="",
                        question_id=question_id,
                        answer="",
                        reference=rubric_reference(rubric),
                        max_marks=rubric["max_marks"],
                        status="SKIPPED_UNCERTAIN_SEGMENTATION",
                        reason=(
                            "A transcript could not be assigned to question IDs safely; "
                            "no mark was fabricated for this question."
                        ),
                    )
                    continue
                marks[student_id][question_id] = 0.0
                append_log(
                    student_id=student_id,
                    submission_id="",
                    question_id=question_id,
                    answer="",
                    reference=rubric_reference(rubric),
                    max_marks=rubric["max_marks"],
                    awarded=0.0,
                    status="NOT_SUBMITTED_ZERO",
                    reason=(
                        "No submission or confidently segmented answer was found; "
                        "counted as 0 for the unattempted question."
                    ),
                )

        verified_sums: Dict[str, float] = defaultdict(float)
        if "teacher_verified" in submissions_columns and "teacher_score" in submissions_columns:
            for student_id, rows in submissions_by_student.items():
                for row in rows:
                    verified = row_value(row, submissions_columns, "teacher_verified", 0)
                    teacher_score = row_value(row, submissions_columns, "teacher_score")
                    if verified and teacher_score is not None:
                        verified_sums[student_id] += float(teacher_score)

        updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        ledger_rows: List[Dict[str, Any]] = []
        for student_id in student_ids:
            scores = marks[student_id]

            def pair_total(question_pair: Sequence[str]) -> Optional[float]:
                values = [scores[question_id] for question_id in question_pair]
                if any(value is None for value in values):
                    return None
                return round(sum(float(value) for value in values), 2)

            q1_total = pair_total(("Q1A", "Q1B"))
            q2_total = pair_total(("Q2A", "Q2B"))
            q3_total = pair_total(("Q3A", "Q3B"))
            if (
                student_id not in unresolved_students
                and q1_total is not None
                and q2_total is not None
                and q3_total is not None
            ):
                breakdown = calculate_ia_total(
                    {question_id: float(scores[question_id]) for question_id in question_ids}
                )
                best_total = max(q1_total, q2_total)
                final_total: Any = breakdown["ia_total"]
            else:
                best_total = (
                    max(q1_total, q2_total)
                    if q1_total is not None and q2_total is not None
                    else None
                )
                final_total = (
                    round(best_total + q3_total, 2)
                    if student_id not in unresolved_students
                    and best_total is not None
                    and q3_total is not None
                    else None
                )

            answer_count = len(answered_questions[student_id])
            if student_id in review_students:
                status = "Needs review"
            elif answer_count == 0:
                status = "No answers"
            elif answer_count < 4:
                status = "Incomplete - review"
            else:
                status = "BERT evaluated"

            row: Dict[str, Any] = {
                "Student USN": excel_safe_text(student_id),
                "Q1 Total": "" if q1_total is None else q1_total,
                "Q2 Total": "" if q2_total is None else q2_total,
                "Best (Q1/Q2)": "" if best_total is None else best_total,
                "Q3 Total": "" if q3_total is None else q3_total,
                "Final IA Total": "" if final_total is None else final_total,
                "Max Marks": breakdown["max_marks"] if (
                    student_id not in unresolved_students
                    and q1_total is not None
                    and q2_total is not None
                    and q3_total is not None
                ) else calculate_ia_total({})["max_marks"],
                "Verified Sum": round(verified_sums.get(student_id, 0.0), 2),
                "Answered": answer_count,
                "Status": status,
                "Last Updated": updated_at,
            }
            for question_id in question_ids:
                score = scores[question_id]
                row[question_id] = "" if score is None else score
            ledger_rows.append(row)

        validate_outputs(ledger_rows, log_rows, question_ids, rubrics, student_ids)
        write_csv_atomic(ledger_path, LEDGER_FIELDS, ledger_rows)
        write_csv_atomic(log_path, LOG_FIELDS, log_rows)

        skipped_answers = sum(
            1
            for row in log_rows
            if str(row["Evaluation Status"]).startswith(("NOT_SUBMITTED", "SKIPPED"))
            or str(row["Evaluation Status"]).startswith("FAILED_SEGMENTATION")
        )
        total_marks = sum(
            float(row["Awarded Marks"])
            for row in log_rows
            if row["Awarded Marks"] not in ("", None)
            and str(row["Evaluation Status"]).startswith("EVALUATED")
        )
        print("\nBERT evaluation summary")
        print("Database: %s" % database_path)
        print("Exam ID: %s" % (exam_id or "(not stored)"))
        print("BERT model: %s" % MODEL_ID)
        print("Total students found: %d" % len(student_ids))
        print("Total students processed: %d" % len(ledger_rows))
        print("Total submissions processed: %d" % len(all_submission_rows))
        print("Selected-exam submissions: %d" % len(submission_rows))
        print("Submissions from other exams skipped: %d" % len(excluded_submission_rows))
        print("Total questions expected: %d (%d student-question slots)" % (
            len(question_ids), len(student_ids) * len(question_ids)
        ))
        print("Total answers detected: %d" % detected_answers)
        print("Total answers evaluated: %d" % evaluated_answers)
        print("Total answers failed: %d" % failed_answers)
        print("Total answers skipped: %d" % skipped_answers)
        print("Total marks generated: %.2f (sum of question-level BERT marks)" % total_marks)
        print("Ledger: %s" % ledger_path)
        print("Evaluation log: %s" % log_path)
        print("\nStudent-wise summary")
        print("Student USN | Answered | Q1 Total | Q2 Total | Best | Q3 Total | Final IA | Status")
        for row in ledger_rows:
            print(
                "%s | %s | %s | %s | %s | %s | %s | %s"
                % (
                    row["Student USN"],
                    row["Answered"],
                    row["Q1 Total"] if row["Q1 Total"] != "" else "—",
                    row["Q2 Total"] if row["Q2 Total"] != "" else "—",
                    row["Best (Q1/Q2)"] if row["Best (Q1/Q2)"] != "" else "—",
                    row["Q3 Total"] if row["Q3 Total"] != "" else "—",
                    row["Final IA Total"] if row["Final IA Total"] != "" else "—",
                    row["Status"],
                )
            )

        exceptions = [
            row
            for row in log_rows
            if str(row["Evaluation Status"]).startswith(
                ("FAILED", "SKIPPED", "NOT_SUBMITTED")
            )
            and row["Evaluation Status"] != "NOT_SUBMITTED_ZERO"
        ]
        if exceptions:
            print("\nAnswers requiring review")
            for row in exceptions:
                print(
                    "%s / %s / submission %s: %s"
                    % (
                        row["Student USN"],
                        row["Question ID"],
                        row["Submission ID"] or "(none)",
                        row["Error/Skip Reason"],
                    )
                )
        return 0 if failed_answers == 0 else 1
    finally:
        connection.close()


def validate_outputs(
    ledger_rows: Sequence[Dict[str, Any]],
    log_rows: Sequence[Dict[str, Any]],
    question_ids: Sequence[str],
    rubrics: Dict[str, Dict[str, Any]],
    student_ids: Sequence[str],
) -> None:
    if len(ledger_rows) != len(student_ids):
        raise EvaluationError("Ledger row count does not match the student count.")
    ledger_ids = [str(row["Student USN"]) for row in ledger_rows]
    if len(set(ledger_ids)) != len(ledger_ids):
        raise EvaluationError("Duplicate student IDs found in the ledger.")
    for row in ledger_rows:
        for question_id in question_ids:
            value = row[question_id]
            if value == "":
                continue
            score = float(value)
            if score < 0 or score > rubrics[question_id]["max_marks"]:
                raise EvaluationError(
                    "Out-of-range ledger mark for %s / %s: %s"
                    % (row["Student USN"], question_id, score)
                )
        if row["Final IA Total"] != "":
            if float(row["Final IA Total"]) > float(row["Max Marks"]):
                raise EvaluationError(
                    "Final IA total exceeds the maximum for %s."
                    % row["Student USN"]
                )
            calculated = calculate_ia_total(
                {question_id: float(row[question_id]) for question_id in question_ids}
            )
            if (
                round(float(row["Q1 Total"]), 2) != calculated["q1_total"]
                or round(float(row["Q2 Total"]), 2) != calculated["q2_total"]
                or round(float(row["Q3 Total"]), 2) != calculated["q3_total"]
                or round(float(row["Best (Q1/Q2)"]), 2)
                != max(calculated["q1_total"], calculated["q2_total"])
                or round(float(row["Final IA Total"]), 2) != calculated["ia_total"]
            ):
                raise EvaluationError(
                    "IA totals do not match the project's choice rule for %s."
                    % row["Student USN"]
                )
    if any(row["Question ID"] is None for row in log_rows):
        raise EvaluationError("An evaluation log entry has no question identifier.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a SQLite exam database and export a one-row-per-student "
            "BERT marks ledger plus an answer-level evaluation log."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=(
            "Input SQLite database. Defaults to ~/Downloads/grading_system.db, "
            "then backend/grading_system.db."
        ),
    )
    parser.add_argument(
        "--exam-id",
        default=None,
        help="Required only when the database contains more than one exam.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Directory for answer1.csv and bert_evaluation_log.csv (default: project root).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        database_path = select_database(args.db)
        output_dir = args.output_dir.expanduser().resolve()
        return evaluate_database(
            database_path=database_path,
            exam_id_arg=args.exam_id,
            ledger_path=output_dir / "answer1.csv",
            log_path=output_dir / "bert_evaluation_log.csv",
        )
    except Exception as exc:
        print(
            "BERT evaluation stopped: %s: %s" % (type(exc).__name__, exc),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
