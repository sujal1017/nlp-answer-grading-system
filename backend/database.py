import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parent
DB_NAME = str(BACKEND_DIR / "grading_system.db")
DATA_DIR = BACKEND_DIR / "data"
TRAINING_LOG = DATA_DIR / "training_dataset.jsonl"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


DEFAULT_SDMCET_QUESTIONS = [
    {
        "question_id": "Q1A",
        "question_text": "Define research and explain its major objectives",
        "max_marks": 5,
        "key_points": [
            {"point": "Clear and accurate definition of research", "weight": 1.25},
            {"point": "Primary exploratory, descriptive, and diagnostic objectives", "weight": 1.25},
            {"point": "Academic and practical significance of research", "weight": 1.25},
            {"point": "Systematic nature and sequential methodology", "weight": 1.25},
        ],
    },
    {
        "question_id": "Q1B",
        "question_text": "Describe the systematic procedure involved in reviewing the literature",
        "max_marks": 5,
        "key_points": [
            {"point": "Identification of sources and domain keywords", "weight": 1.25},
            {"point": "Critical analysis and summarization of past work", "weight": 1.25},
            {"point": "Identifying research gaps and inconsistencies", "weight": 1.25},
            {"point": "Thematic synthesis and structuring of the review report", "weight": 1.25},
        ],
    },
    {
        "question_id": "Q2A",
        "question_text": "Classify and explain the different types of research with examples",
        "max_marks": 5,
        "key_points": [
            {"point": "Fundamental vs applied research distinction", "weight": 1.25},
            {"point": "Qualitative vs quantitative methodologies", "weight": 1.25},
            {"point": "Descriptive vs analytical research characteristics", "weight": 1.25},
            {"point": "Concrete real-world examples for each category", "weight": 1.25},
        ],
    },
    {
        "question_id": "Q2B",
        "question_text": "Discuss the essential features of a good research design",
        "max_marks": 5,
        "key_points": [
            {"point": "Clear formulation of the research problem and blueprint", "weight": 1.25},
            {"point": "Reliability, validity, and objectivity of procedures", "weight": 1.25},
            {"point": "Control of extraneous variables and experimental variance", "weight": 1.25},
            {"point": "Feasibility, sampling plan, and resource economy", "weight": 1.25},
        ],
    },
    {
        "question_id": "Q3A",
        "question_text": "Explain the various techniques used for defining a research problem",
        "max_marks": 5,
        "key_points": [
            {"point": "Identifying domain interest and broad concept exploration", "weight": 1.25},
            {"point": "Reviewing existing literature survey and finding gaps", "weight": 1.25},
            {"point": "Narrowing down keywords and setting specific objectives", "weight": 1.25},
            {"point": "Feasibility, preliminary surveys, and problem statement drafting", "weight": 1.25},
        ],
    },
    {
        "question_id": "Q3B",
        "question_text": "Discuss the important concepts associated with research design",
        "max_marks": 5,
        "key_points": [
            {"point": "Dependent and independent variables formulation", "weight": 1.25},
            {"point": "Extraneous and confounding variables control", "weight": 1.25},
            {"point": "Experimental vs control groups and hypotheses", "weight": 1.25},
            {"point": "Testing adequacy and grouping parameters", "weight": 1.25},
        ],
    },
]


def update_exam_textbook(exam_id: str, filename: str):
    """Updates textbook_filename for an exam."""
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE exams SET textbook_filename=? WHERE exam_id=?", (filename, eid))
    conn.commit()
    conn.close()


def sync_student_result(exam_id: str, student_id: str, conn: Optional[sqlite3.Connection] = None) -> Dict:
    """Auto-Sync Scorecard Function:
    Queries all rows in submissions for (exam_id, student_id).
    Uses teacher_score if teacher_verified == 1, else uses ai_score for each question (Q1A–Q3B).
    Computes q1_total = q1a + q1b, q2_total = q2a + q2b, q3_total = q3a + q3b,
    and final_ia_total = max(q1_total, q2_total) + q3_total.
    Upserts the complete scorecard into student_results.
    """
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    cursor = conn.cursor()
    cursor.execute(
        "SELECT question_id, ai_score, teacher_score, teacher_verified, status FROM submissions WHERE exam_id=? AND student_id=?",
        (eid, sid),
    )
    rows = cursor.fetchall()
    sub_map = {}
    verified_count = 0
    for r in rows:
        qid = r["question_id"].upper()
        t_score = r["teacher_score"]
        t_ver = r["teacher_verified"]
        ai_sc = r["ai_score"]
        if t_ver == 1 and t_score is not None:
            score = float(t_score)
            verified_count += 1
        else:
            score = float(ai_sc or 0.0)
        sub_map[qid] = score

    q1a = sub_map.get("Q1A", 0.0)
    q1b = sub_map.get("Q1B", 0.0)
    q2a = sub_map.get("Q2A", 0.0)
    q2b = sub_map.get("Q2B", 0.0)
    q3a = sub_map.get("Q3A", 0.0)
    q3b = sub_map.get("Q3B", 0.0)

    q1_total = round(q1a + q1b, 2)
    q2_total = round(q2a + q2b, 2)
    q3_total = round(q3a + q3b, 2)
    final_ia_total = round(max(q1_total, q2_total) + q3_total, 2)
    total_answered = len(rows)

    status = "evaluated"
    if total_answered > 0 and verified_count == total_answered:
        status = "verified"
    elif total_answered == 0:
        status = "not_started"

    cursor.execute(
        """
        INSERT INTO student_results (
            exam_id, student_id,
            q1a_score, q1b_score, q2a_score, q2b_score, q3a_score, q3b_score,
            q1_total, q2_total, q3_total, final_ia_total, max_ia_marks,
            verified_count, total_questions_answered, status, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 20.0, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(exam_id, student_id) DO UPDATE SET
            q1a_score = excluded.q1a_score,
            q1b_score = excluded.q1b_score,
            q2a_score = excluded.q2a_score,
            q2b_score = excluded.q2b_score,
            q3a_score = excluded.q3a_score,
            q3b_score = excluded.q3b_score,
            q1_total = excluded.q1_total,
            q2_total = excluded.q2_total,
            q3_total = excluded.q3_total,
            final_ia_total = excluded.final_ia_total,
            max_ia_marks = excluded.max_ia_marks,
            verified_count = excluded.verified_count,
            total_questions_answered = excluded.total_questions_answered,
            status = excluded.status,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            eid, sid,
            q1a, q1b, q2a, q2b, q3a, q3b,
            q1_total, q2_total, q3_total, final_ia_total,
            verified_count, total_answered, status,
        ),
    )
    conn.commit()

    scorecard = {
        "exam_id": eid,
        "student_id": sid,
        "q1a_score": q1a,
        "q1b_score": q1b,
        "q2a_score": q2a,
        "q2b_score": q2b,
        "q3a_score": q3a,
        "q3b_score": q3b,
        "q1_total": q1_total,
        "q2_total": q2_total,
        "q3_total": q3_total,
        "final_ia_total": final_ia_total,
        "max_ia_marks": 20.0,
        "verified_count": verified_count,
        "total_questions_answered": total_answered,
        "status": status,
    }

    if close_conn:
        conn.close()

    return scorecard


def get_student_results(exam_id: str) -> List[Dict]:
    """Returns all rows from student_results for the exam ordered by student_id ASC."""
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM student_results WHERE exam_id=? ORDER BY student_id ASC",
        (eid,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def seed_default_exam(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM exams")
    count = cursor.fetchone()[0]
    if count == 0:
        exam_id = "IA1_RM_2026"
        exam_title = "22URMK506 – Research Methodology & IPR"
        raw_text = "\n".join(
            f"{q['question_id']} {q['question_text']} {q['max_marks']} Marks"
            for q in DEFAULT_SDMCET_QUESTIONS
        )
        cursor.execute(
            """
            INSERT INTO exams (exam_id, exam_title, subject, qp_image_paths, raw_qp_text)
            VALUES (?, ?, 'Research Methodology & IPR', '[]', ?)
            """,
            (exam_id, exam_title, raw_text),
        )

        for q in DEFAULT_SDMCET_QUESTIONS:
            cursor.execute(
                """
                INSERT INTO rubrics (exam_id, question_id, question_text, max_marks, key_points)
                VALUES (?, ?, ?, ?, ?)
                """,
                (exam_id, q["question_id"], q["question_text"], q["max_marks"], json.dumps(q["key_points"])),
            )

        # Check if sample booklet images exist for student 2SD23CS082
        sample_student_dir = DATA_DIR / "booklets" / exam_id / "2SD23CS082"
        if sample_student_dir.exists():
            image_rel_paths = [
                f"booklets/{exam_id}/2SD23CS082/page_{i}.png" for i in range(1, 9)
                if (sample_student_dir / f"page_{i}.png").exists()
            ]

            sample_subs = [
                (
                    "Q1A",
                    "Research is defined as the systematic and sequential process of defining, analyzing, & interpreting the given question or problem.\nRe = 'Again & again'\nSearch = 'Analyzing & interpreting the answer to the problem'\n\nThe major objectives of research include:\n1. To define the problem statement\n2. To gain basic knowledge and theories\n3. To find research gaps\n4. To analyze depth in a domain\n5. For academic advancement\n6. To help society with evidence-based decisions.",
                    ["Clear and accurate definition of research", "Primary exploratory, descriptive, and diagnostic objectives", "Academic and practical significance of research", "Systematic nature and sequential methodology"],
                    [],
                    5.0,
                    "Student provides comprehensive definition and thoroughly explains exploratory, descriptive, and academic objectives.",
                    5.0,
                    1,
                ),
                (
                    "Q1B",
                    "Systematic Procedure of Reviewing Literature:\n1. Define the Problem Statement: Clearly specify scope.\n2. Globalize the Problem: Understand stakeholder impacts.\n3. Collect Past Submitted Literature: Gather research articles.\n4. Search for Key Contents Regarding Domain: Concepts and tools.\n5. Analyze Collected Literature: Summarize main findings and note gaps.\n6. Prepare Pilot Document: Formulate research questions and analysis plan.\n7. Collect Data: Surveys, experiments, observations.\n8. Analyze Data: Use statistical tools and synthesize insights.",
                    ["Identification of sources and domain keywords", "Critical analysis and summarization of past work", "Identifying research gaps and inconsistencies", "Thematic synthesis and structuring of the review report"],
                    [],
                    5.0,
                    "Detailed 8-step breakdown of literature review methodology with clear critical analysis and gap identification.",
                    5.0,
                    1,
                ),
                (
                    "Q3A",
                    "Techniques Used for Defining Research Problem:\n1. Basic Ideal Interests in Specific Domain: Identify primary interest.\n2. Broadly and Widely Go Through Concepts/Domains: Explore theories.\n3. Identify Gaps: Locate less-discussed areas needing research.\n4. Narrow Down Search: Refine keywords and specific focus.\n5. Select the Specific Problem: Precise formulation.\n6. Literature Survey: Review trends and methodologies.\n7. Determine Specific Objectives: Clearly defined outcomes.\n8. Collect Multi-Platform Data: Journals, databases, interviews.\n9. Synthesize and Analyze: Structure conclusions and check feasibility.",
                    ["Identifying domain interest and broad concept exploration", "Reviewing existing literature survey and finding gaps", "Narrowing down keywords and setting specific objectives", "Feasibility, preliminary surveys, and problem statement drafting"],
                    [],
                    4.5,
                    "Excellent structured walkthrough of problem definition techniques; covers keyword narrowing and objectives.",
                    4.5,
                    1,
                ),
                (
                    "Q3B",
                    "Concepts Associated with Research Design:\n1. Dependent Variables: Factors or variables directly affecting or measuring the outcome of an experiment.\n2. Independent Variables: Variables that are manipulated to observe effects on dependent outcomes.\n3. Extraneous Variables: External factors that can affect experimental validity if not controlled.\n4. Testing and Adequacy: Verification that sample variables are representative.\n5. Hypotheses: Formulating testable assumptions regarding design parameters.\n6. Grouping: Structuring experimental and control conditions.",
                    ["Dependent and independent variables formulation", "Extraneous and confounding variables control", "Experimental vs control groups and hypotheses", "Testing adequacy and grouping parameters"],
                    [],
                    4.5,
                    "Covers all four major concepts: dependent/independent variables, extraneous controls, hypothesis formulation, and test adequacy.",
                    4.5,
                    1,
                ),
            ]

            full_sample_text = "\n\n--- Page 1 ---\n\n".join(
                f"Question {qid}\n{text}" for qid, text, *_ in sample_subs
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO booklets (
                    exam_id, student_id, image_paths, full_transcribed_text
                ) VALUES (?, '2SD23CS082', ?, ?)
                """,
                (exam_id, json.dumps(image_rel_paths), full_sample_text),
            )

            for qid, text, matched, missing, ai_sc, fb, tch_sc, tch_vf in sample_subs:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO submissions (
                        exam_id, student_id, question_id, image_paths, transcribed_text,
                        matched_points, missing_points, ai_score, ai_feedback, teacher_score,
                        teacher_verified, status
                    ) VALUES (?, '2SD23CS082', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Verified')
                    """,
                    (
                        exam_id,
                        qid,
                        json.dumps(image_rel_paths),
                        text,
                        json.dumps(matched),
                        json.dumps(missing),
                        ai_sc,
                        fb,
                        tch_sc,
                        tch_vf,
                    ),
                )
        conn.commit()
        sync_student_result(exam_id, "2SD23CS082", conn=conn)
    else:
        # If exam exists, verify if rubrics need upgrading to 4 atomic criteria
        cursor.execute("SELECT key_points FROM rubrics WHERE exam_id='IA1_RM_2026' AND question_id='Q1A'")
        row = cursor.fetchone()
        if row:
            try:
                pts = json.loads(row[0])
                if len(pts) < 4:
                    for q in DEFAULT_SDMCET_QUESTIONS:
                        cursor.execute(
                            """
                            UPDATE rubrics SET question_text=?, max_marks=?, key_points=?
                            WHERE exam_id='IA1_RM_2026' AND question_id=?
                            """,
                            (q["question_text"], q["max_marks"], json.dumps(q["key_points"]), q["question_id"]),
                        )
                    conn.commit()
            except Exception:
                pass


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "booklets").mkdir(exist_ok=True)
    (DATA_DIR / "exams").mkdir(exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS exams (
            exam_id TEXT PRIMARY KEY,
            exam_title TEXT NOT NULL,
            subject TEXT DEFAULT 'Research Methodology & IPR',
            qp_image_paths TEXT,
            raw_qp_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rubrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            question_text TEXT NOT NULL,
            max_marks INTEGER NOT NULL,
            key_points TEXT NOT NULL,
            UNIQUE(exam_id, question_id),
            FOREIGN KEY(exam_id) REFERENCES exams(exam_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS booklets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            image_paths TEXT NOT NULL,
            full_transcribed_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(exam_id, student_id),
            FOREIGN KEY(exam_id) REFERENCES exams(exam_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            image_paths TEXT NOT NULL,
            transcribed_text TEXT,
            matched_points TEXT,
            missing_points TEXT,
            ai_score REAL DEFAULT 0.0,
            ai_feedback TEXT,
            teacher_score REAL,
            teacher_verified INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Uploaded',
            UNIQUE(exam_id, student_id, question_id),
            FOREIGN KEY(exam_id) REFERENCES exams(exam_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            q1a_score REAL DEFAULT 0.0,
            q1b_score REAL DEFAULT 0.0,
            q2a_score REAL DEFAULT 0.0,
            q2b_score REAL DEFAULT 0.0,
            q3a_score REAL DEFAULT 0.0,
            q3b_score REAL DEFAULT 0.0,
            q1_total REAL DEFAULT 0.0,
            q2_total REAL DEFAULT 0.0,
            q3_total REAL DEFAULT 0.0,
            final_ia_total REAL DEFAULT 0.0,
            max_ia_marks REAL DEFAULT 20.0,
            verified_count INTEGER DEFAULT 0,
            total_questions_answered INTEGER DEFAULT 0,
            status TEXT DEFAULT 'evaluated',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(exam_id, student_id)
        )
        """
    )

    # Safe ALTER TABLE migration for textbook_filename on exams table
    cursor.execute("PRAGMA table_info(exams)")
    exam_cols = [r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in cursor.fetchall()]
    if "textbook_filename" not in exam_cols:
        cursor.execute("ALTER TABLE exams ADD COLUMN textbook_filename TEXT DEFAULT ''")

    conn.commit()
    seed_default_exam(conn)

    # Ensure any existing student submissions have synced scorecards
    cursor.execute("SELECT DISTINCT exam_id, student_id FROM submissions")
    existing_students = cursor.fetchall()
    for s_row in existing_students:
        sync_student_result(s_row["exam_id"], s_row["student_id"], conn=conn)

    conn.close()


def create_exam(exam_id: str, exam_title: str, qp_image_paths: Optional[List[str]] = None, raw_qp_text: str = ""):
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO exams (exam_id, exam_title, qp_image_paths, raw_qp_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(exam_id) DO UPDATE SET
            exam_title = excluded.exam_title,
            qp_image_paths = excluded.qp_image_paths,
            raw_qp_text = excluded.raw_qp_text
        """,
        (eid, exam_title, json.dumps(qp_image_paths or []), raw_qp_text),
    )
    conn.commit()
    conn.close()


def get_all_exams() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exams ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_exam(exam_id: str) -> Optional[Dict]:
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exams WHERE exam_id = ?", (eid,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_exam_rubrics(exam_id: str):
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM rubrics WHERE exam_id = ?", (eid,))
    conn.commit()
    conn.close()


def save_question_rubric(exam_id: str, question_id: str, question_text: str, max_marks: int, key_points: List):
    eid = exam_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO rubrics (exam_id, question_id, question_text, max_marks, key_points)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(exam_id, question_id) DO UPDATE SET
            question_text = excluded.question_text,
            max_marks = excluded.max_marks,
            key_points = excluded.key_points
        """,
        (eid, qid, question_text, max_marks, json.dumps(key_points)),
    )
    conn.commit()
    conn.close()


def get_exam_rubrics(exam_id: str) -> List[Dict]:
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rubrics WHERE exam_id = ? ORDER BY question_id", (eid,))
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        item = dict(r)
        item["key_points"] = json.loads(item["key_points"])
        result.append(item)
    return result


def save_raw_booklet(exam_id: str, student_id: str, image_paths: List[str], full_transcribed_text: str):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO booklets (exam_id, student_id, image_paths, full_transcribed_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(exam_id, student_id) DO UPDATE SET
            image_paths = excluded.image_paths,
            full_transcribed_text = excluded.full_transcribed_text
        """,
        (eid, sid, json.dumps(image_paths), full_transcribed_text),
    )
    conn.commit()
    conn.close()


def get_raw_booklet(exam_id: str, student_id: str) -> Optional[Dict]:
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM booklets WHERE exam_id=? AND student_id=?",
        (eid, sid),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    item = dict(row)
    item["image_paths"] = json.loads(item["image_paths"]) if item["image_paths"] else []
    return item


def get_exam_booklets(exam_id: str) -> List[Dict]:
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM booklets WHERE exam_id=? ORDER BY student_id ASC",
        (eid,),
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        item = dict(r)
        item["image_paths"] = json.loads(item["image_paths"]) if item["image_paths"] else []
        result.append(item)
    return result


def delete_student_submission(exam_id: str, student_id: str, question_id: str) -> Dict[str, Any]:
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM submissions WHERE exam_id=? AND student_id=? AND question_id=?",
        (eid, sid, qid),
    )
    conn.commit()
    conn.close()
    sync_student_result(eid, sid)

    from chunker import calculate_ia_total
    updated_subs = get_exam_submissions(eid, sid)
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in updated_subs}
    ia_breakdown = calculate_ia_total(scores)

    return {
        "status": "success",
        "submissions": updated_subs,
        "ia_total": ia_breakdown["ia_total"],
        "ia_breakdown": ia_breakdown,
    }


def delete_all_student_submissions(exam_id: str, student_id: str):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM submissions WHERE exam_id=? AND student_id=?",
        (eid, sid),
    )
    conn.commit()
    conn.close()
    sync_student_result(eid, sid)


def save_student_answer(
    exam_id: str,
    student_id: str,
    question_id: str,
    image_paths: List[str],
    transcribed_text: str,
    ai_score: float = 0.0,
):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO submissions (exam_id, student_id, question_id, image_paths, transcribed_text, ai_score, status)
        VALUES (?, ?, ?, ?, ?, ?, 'Transcribed')
        ON CONFLICT(exam_id, student_id, question_id) DO UPDATE SET
            image_paths = excluded.image_paths,
            transcribed_text = excluded.transcribed_text,
            ai_score = excluded.ai_score,
            status = 'Transcribed',
            teacher_verified = 0
        """,
        (eid, sid, qid, json.dumps(image_paths), transcribed_text, ai_score),
    )
    conn.commit()
    conn.close()
    sync_student_result(eid, sid)


def update_evaluation(
    exam_id: str,
    student_id: str,
    question_id: str,
    matched_points: List[str],
    missing_points: List[str],
    ai_score: float,
    ai_feedback: str,
):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE submissions
        SET matched_points = ?,
            missing_points = ?,
            ai_score = ?,
            ai_feedback = ?,
            teacher_score = ?,
            status = 'Evaluated'
        WHERE exam_id = ? AND student_id = ? AND question_id = ?
        """,
        (
            json.dumps(matched_points),
            json.dumps(missing_points),
            ai_score,
            ai_feedback,
            ai_score,
            eid,
            sid,
            qid,
        ),
    )
    conn.commit()
    conn.close()
    sync_student_result(eid, sid)


def verify_submission(exam_id: str, student_id: str, question_id: str, teacher_score: float, ai_feedback: str):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE submissions
        SET teacher_score = ?,
            ai_feedback = ?,
            teacher_verified = 1,
            status = 'Verified'
        WHERE exam_id = ? AND student_id = ? AND question_id = ?
        """,
        (teacher_score, ai_feedback, eid, sid, qid),
    )
    conn.commit()
    conn.close()
    sync_student_result(eid, sid)


def get_exam_submissions(exam_id: str, student_id: Optional[str] = None) -> List[Dict]:
    eid = exam_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    if student_id:
        sid = student_id.strip().upper()
        cursor.execute(
            "SELECT * FROM submissions WHERE exam_id = ? AND student_id = ? ORDER BY question_id",
            (eid, sid),
        )
    else:
        cursor.execute(
            "SELECT * FROM submissions WHERE exam_id = ? ORDER BY student_id, question_id",
            (eid,),
        )
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        item = dict(r)
        item["image_paths"] = json.loads(item["image_paths"]) if item["image_paths"] else []
        item["matched_points"] = json.loads(item["matched_points"]) if item["matched_points"] else []
        item["missing_points"] = json.loads(item["missing_points"]) if item["missing_points"] else []
        item["ai_score"] = float(item["ai_score"]) if item.get("ai_score") is not None else 0.0
        result.append(item)
    return result


def update_rubric_weights_from_feedback(
    exam_id: str,
    student_id: str,
    question_id: str,
    teacher_score: float,
    learning_rate: float = 0.25,
):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND question_id=?",
        (eid, sid, qid),
    )
    sub = cursor.fetchone()

    cursor.execute(
        "SELECT * FROM rubrics WHERE exam_id=? AND question_id=?",
        (eid, qid),
    )
    rub = cursor.fetchone()

    if not sub or not rub:
        conn.close()
        return None

    matched = json.loads(sub["matched_points"]) if sub["matched_points"] else []
    key_points = json.loads(rub["key_points"])
    max_marks = float(rub["max_marks"])
    ai_score = float(sub["ai_score"] or 0.0)

    if key_points and isinstance(key_points[0], str):
        default_w = round(max_marks / len(key_points), 2)
        key_points = [{"point": pt, "weight": default_w} for pt in key_points]

    error = teacher_score - ai_score
    if abs(error) > 0.01 and matched and key_points:
        adjustment = (learning_rate * error) / max(len(matched), 1)
        for item in key_points:
            if item["point"] in matched:
                item["weight"] = round(max(0.25, min(max_marks, item["weight"] + adjustment)), 3)

        cursor.execute(
            "UPDATE rubrics SET key_points=? WHERE exam_id=? AND question_id=?",
            (json.dumps(key_points), eid, qid),
        )
        conn.commit()

    conn.close()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    training_entry = {
        "exam_id": eid,
        "student_id": sid,
        "question_id": qid,
        "question_text": rub["question_text"],
        "student_answer": sub["transcribed_text"],
        "matched_points": matched,
        "teacher_verified_score": teacher_score,
        "feedback": sub["ai_feedback"],
    }
    with open(TRAINING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(training_entry) + "\n")

    return key_points


def get_verified_examples(exam_id: str, question_id: str, limit: int = 2) -> List[Dict]:
    eid = exam_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT transcribed_text, teacher_score, ai_feedback
        FROM submissions
        WHERE exam_id=? AND question_id=? AND teacher_verified=1
        ORDER BY submission_id DESC LIMIT ?
        """,
        (eid, qid, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_kpi_counts(exam_id: Optional[str] = None) -> Dict:
    conn = get_connection()
    cursor = conn.cursor()
    if exam_id and exam_id.upper() not in {"", "NO EXAMS CREATED"}:
        eid = exam_id.strip().upper()
        cursor.execute("SELECT COUNT(*) FROM rubrics WHERE exam_id=?", (eid,))
        questions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT student_id) FROM booklets WHERE exam_id=?", (eid,))
        booklets = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM submissions WHERE exam_id=?", (eid,))
        answers = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM submissions WHERE exam_id=? AND teacher_verified=1", (eid,))
        verified = cursor.fetchone()[0]
    else:
        questions = booklets = answers = verified = 0
    conn.close()
    return {
        "questions": questions,
        "booklets": booklets,
        "answers": answers,
        "verified": verified,
    }


def assign_segment_to_question(
    exam_id: str,
    student_id: str,
    question_id: str,
    start_char: Optional[int] = None,
    end_char: Optional[int] = None,
    custom_text: Optional[str] = None,
    remove_from_unassigned: bool = True,
) -> Dict[str, Any]:
    """
    Assign a text range (or custom edited text) to a specific question rubric:
      - Explicitly supports start_char == 0 (e.g. when student skips Q1 and starts on Q2A).
      - Immediately evaluates answer against target rubric using EduEval.
      - Upserts into submissions and recalculates the SDMCET IA Choice Rule total.
    """
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM booklets WHERE exam_id=? AND student_id=?", (eid, sid))
    booklet_row = cursor.fetchone()
    full_text = booklet_row["full_transcribed_text"] if booklet_row else ""
    image_paths_raw = booklet_row["image_paths"] if booklet_row else "[]"

    if custom_text is not None and custom_text.strip():
        ans_text = custom_text.strip()
    elif start_char is not None:
        s_idx = max(0, int(start_char))
        e_idx = len(full_text) if end_char is None else min(len(full_text), int(end_char))
        ans_text = full_text[s_idx:e_idx].strip()
    else:
        ans_text = full_text.strip()

    if not ans_text:
        conn.close()
        raise ValueError("Cannot assign an empty text segment.")

    cursor.execute("SELECT * FROM rubrics WHERE exam_id=? AND question_id=?", (eid, qid))
    rub = cursor.fetchone()
    if not rub:
        conn.close()
        raise ValueError(f"Rubric for question '{qid}' not found for exam '{eid}'")

    from scorer import evaluate_with_edueval
    from chunker import calculate_ia_total

    past_examples = get_verified_examples(eid, qid, limit=2)
    eval_res = evaluate_with_edueval(
        question_text=rub["question_text"],
        max_marks=float(rub["max_marks"]),
        key_points=json.loads(rub["key_points"]),
        student_answer=ans_text,
        verified_examples=past_examples,
    )

    cursor.execute(
        """
        INSERT INTO submissions (
            exam_id, student_id, question_id, image_paths, transcribed_text,
            matched_points, missing_points, ai_score, ai_feedback, teacher_score,
            teacher_verified, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'Evaluated')
        ON CONFLICT(exam_id, student_id, question_id) DO UPDATE SET
            image_paths = excluded.image_paths,
            transcribed_text = excluded.transcribed_text,
            matched_points = excluded.matched_points,
            missing_points = excluded.missing_points,
            ai_score = excluded.ai_score,
            ai_feedback = excluded.ai_feedback,
            teacher_score = excluded.teacher_score,
            teacher_verified = 0,
            status = 'Evaluated'
        """,
        (
            eid,
            sid,
            qid,
            image_paths_raw,
            ans_text,
            json.dumps(eval_res["matched_points"]),
            json.dumps(eval_res["missing_points"]),
            eval_res["ai_score"],
            eval_res["ai_feedback"],
            eval_res["ai_score"],
        ),
    )
    conn.commit()
    conn.close()
    sync_student_result(eid, sid)

    updated_subs = get_exam_submissions(eid, sid)
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in updated_subs}
    ia_breakdown = calculate_ia_total(scores)

    raw_booklet = get_raw_booklet(eid, sid)
    return {
        "status": "success",
        "booklet": raw_booklet,
        "submissions": updated_subs,
        "ia_total": ia_breakdown["ia_total"],
        "ia_breakdown": ia_breakdown,
    }


def split_at_cursor(
    exam_id: str,
    student_id: str,
    source_question_id: str,
    cursor_index: int,
    target_question_id: str,
) -> Dict[str, Any]:
    """
    Split an existing assigned answer at cursor_index:
      - If cursor_index == 0: reassigns the entire text block from source_question_id
        to target_question_id, deletes source_question_id, and grades target_question_id.
      - If cursor_index > 0: keeps text[:cursor_index] in source_question_id, moves
        text[cursor_index:] to target_question_id, and evaluates both with evaluate_with_edueval().
    """
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    source_qid = source_question_id.strip().upper()
    target_qid = target_question_id.strip().upper()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND question_id=?",
        (eid, sid, source_qid),
    )
    sub = cursor.fetchone()
    if not sub:
        conn.close()
        raise ValueError(f"Submission '{source_qid}' not found for student '{sid}'")

    full_text = sub["transcribed_text"] or ""
    image_paths_raw = sub["image_paths"] or "[]"

    cursor.execute("SELECT * FROM rubrics WHERE exam_id=? AND question_id=?", (eid, target_qid))
    target_rub = cursor.fetchone()
    if not target_rub:
        conn.close()
        raise ValueError(f"Rubric for target question '{target_qid}' not found")

    from scorer import evaluate_with_edueval
    from chunker import calculate_ia_total

    if int(cursor_index) == 0:
        eval_target = evaluate_with_edueval(
            question_text=target_rub["question_text"],
            max_marks=float(target_rub["max_marks"]),
            key_points=json.loads(target_rub["key_points"]),
            student_answer=full_text,
            verified_examples=get_verified_examples(eid, target_qid, limit=2),
        )

        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(
                "DELETE FROM submissions WHERE exam_id=? AND student_id=? AND question_id=?",
                (eid, sid, source_qid),
            )
            cursor.execute(
                """
                INSERT INTO submissions (
                    exam_id, student_id, question_id, image_paths, transcribed_text,
                    matched_points, missing_points, ai_score, ai_feedback, teacher_score,
                    teacher_verified, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'Evaluated')
                ON CONFLICT(exam_id, student_id, question_id) DO UPDATE SET
                    image_paths = excluded.image_paths,
                    transcribed_text = excluded.transcribed_text,
                    matched_points = excluded.matched_points,
                    missing_points = excluded.missing_points,
                    ai_score = excluded.ai_score,
                    ai_feedback = excluded.ai_feedback,
                    teacher_score = excluded.teacher_score,
                    teacher_verified = 0,
                    status = 'Evaluated'
                """,
                (
                    eid, sid, target_qid, image_paths_raw, full_text,
                    json.dumps(eval_target["matched_points"]),
                    json.dumps(eval_target["missing_points"]),
                    eval_target["ai_score"],
                    eval_target["ai_feedback"],
                    eval_target["ai_score"],
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise
    else:
        c_idx = max(0, min(len(full_text), int(cursor_index)))
        text1 = full_text[:c_idx].strip()
        text2 = full_text[c_idx:].strip()

        cursor.execute("SELECT * FROM rubrics WHERE exam_id=? AND question_id=?", (eid, source_qid))
        source_rub = cursor.fetchone()
        if not source_rub:
            conn.close()
            raise ValueError(f"Rubric for source question '{source_qid}' not found")

        eval1 = evaluate_with_edueval(
            question_text=source_rub["question_text"],
            max_marks=float(source_rub["max_marks"]),
            key_points=json.loads(source_rub["key_points"]),
            student_answer=text1,
            verified_examples=get_verified_examples(eid, source_qid, limit=2),
        )
        eval2 = evaluate_with_edueval(
            question_text=target_rub["question_text"],
            max_marks=float(target_rub["max_marks"]),
            key_points=json.loads(target_rub["key_points"]),
            student_answer=text2,
            verified_examples=get_verified_examples(eid, target_qid, limit=2),
        )

        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute(
                """
                UPDATE submissions
                SET transcribed_text = ?,
                    matched_points = ?,
                    missing_points = ?,
                    ai_score = ?,
                    ai_feedback = ?,
                    teacher_score = ?,
                    teacher_verified = 0,
                    status = 'Evaluated'
                WHERE exam_id=? AND student_id=? AND question_id=?
                """,
                (
                    text1,
                    json.dumps(eval1["matched_points"]),
                    json.dumps(eval1["missing_points"]),
                    eval1["ai_score"],
                    eval1["ai_feedback"],
                    eval1["ai_score"],
                    eid, sid, source_qid,
                ),
            )
            cursor.execute(
                """
                INSERT INTO submissions (
                    exam_id, student_id, question_id, image_paths, transcribed_text,
                    matched_points, missing_points, ai_score, ai_feedback, teacher_score,
                    teacher_verified, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'Evaluated')
                ON CONFLICT(exam_id, student_id, question_id) DO UPDATE SET
                    image_paths = excluded.image_paths,
                    transcribed_text = excluded.transcribed_text,
                    matched_points = excluded.matched_points,
                    missing_points = excluded.missing_points,
                    ai_score = excluded.ai_score,
                    ai_feedback = excluded.ai_feedback,
                    teacher_score = excluded.teacher_score,
                    teacher_verified = 0,
                    status = 'Evaluated'
                """,
                (
                    eid, sid, target_qid, image_paths_raw, text2,
                    json.dumps(eval2["matched_points"]),
                    json.dumps(eval2["missing_points"]),
                    eval2["ai_score"],
                    eval2["ai_feedback"],
                    eval2["ai_score"],
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise

    conn.close()
    sync_student_result(eid, sid)

    updated_subs = get_exam_submissions(eid, sid)
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in updated_subs}
    ia_breakdown = calculate_ia_total(scores)

    return {
        "status": "success",
        "submissions": updated_subs,
        "ia_total": ia_breakdown["ia_total"],
        "ia_breakdown": ia_breakdown,
    }


def merge_and_reassign_submissions(
    exam_id: str,
    student_id: str,
    source_question_ids: List[str],
    target_question_id: str,
) -> Dict[str, Any]:
    """
    Merge multiple fragmented submission chunks into a unified target question submission:
      1. Fetch source submissions ordered by submission_id to preserve page order.
      2. Concatenate transcribed_text with \\n\\n.
      3. Union image paths.
      4. In a single SQLite transaction, delete all source rows and insert/replace the unified target row.
      5. Re-evaluate against target rubric using EduEval.
      6. Return updated student submissions and recalculated SDMCET IA total.
    """
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    target_qid = target_question_id.strip().upper()
    source_qids = [q.strip().upper() for q in source_question_ids if q.strip()]

    if not source_qids:
        raise ValueError("At least one source question ID is required for merge.")

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join(["?"] * len(source_qids))
    cursor.execute(
        f"SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND question_id IN ({placeholders}) ORDER BY submission_id ASC",
        [eid, sid] + source_qids,
    )
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        raise ValueError(f"No submissions found for sources {source_qids} (student: {sid}, exam: {eid})")

    text_parts = [r["transcribed_text"].strip() for r in rows if r["transcribed_text"] and r["transcribed_text"].strip()]
    unified_text = "\n\n".join(text_parts)

    combined_images = []
    for r in rows:
        imgs = json.loads(r["image_paths"]) if r["image_paths"] else []
        for img in imgs:
            if img not in combined_images:
                combined_images.append(img)

    cursor.execute("SELECT * FROM rubrics WHERE exam_id=? AND question_id=?", (eid, target_qid))
    target_rubric = cursor.fetchone()
    if not target_rubric:
        conn.close()
        raise ValueError(f"Target question rubric '{target_qid}' not found for exam '{eid}'")

    target_key_points = json.loads(target_rubric["key_points"])
    target_max_marks = float(target_rubric["max_marks"])
    target_q_text = target_rubric["question_text"]

    from scorer import evaluate_with_edueval
    from chunker import calculate_ia_total

    past_examples = get_verified_examples(eid, target_qid, limit=2)
    eval_res = evaluate_with_edueval(
        question_text=target_q_text,
        max_marks=target_max_marks,
        key_points=target_key_points,
        student_answer=unified_text,
        verified_examples=past_examples,
    )

    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute(
            f"DELETE FROM submissions WHERE exam_id=? AND student_id=? AND question_id IN ({placeholders})",
            [eid, sid] + source_qids,
        )
        if target_qid not in source_qids:
            cursor.execute(
                "DELETE FROM submissions WHERE exam_id=? AND student_id=? AND question_id=?",
                (eid, sid, target_qid),
            )
        cursor.execute(
            """
            INSERT INTO submissions (
                exam_id, student_id, question_id, image_paths, transcribed_text,
                matched_points, missing_points, ai_score, ai_feedback, teacher_score,
                teacher_verified, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'Evaluated')
            """,
            (
                eid,
                sid,
                target_qid,
                json.dumps(combined_images),
                unified_text,
                json.dumps(eval_res["matched_points"]),
                json.dumps(eval_res["missing_points"]),
                eval_res["ai_score"],
                eval_res["ai_feedback"],
                eval_res["ai_score"],
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise

    conn.close()
    sync_student_result(eid, sid)

    updated_subs = get_exam_submissions(eid, sid)
    scores = {r["question_id"]: float(r["teacher_score"] or r["ai_score"] or 0.0) for r in updated_subs}
    ia_breakdown = calculate_ia_total(scores)

    return {
        "status": "success",
        "submissions": updated_subs,
        "ia_breakdown": ia_breakdown,
    }


def split_submission(
    exam_id: str,
    student_id: str,
    source_question_id: str,
    split_index: int,
    new_question_id: str,
) -> Dict[str, Any]:
    """Delegates to split_at_cursor."""
    return split_at_cursor(
        exam_id=exam_id,
        student_id=student_id,
        source_question_id=source_question_id,
        cursor_index=split_index,
        target_question_id=new_question_id,
    )


if __name__ == "__main__":
    init_db()
    print(f"[+] Database '{DB_NAME}' initialized.")
