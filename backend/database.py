import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

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
        CREATE TABLE IF NOT EXISTS submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            image_paths TEXT NOT NULL,
            transcribed_text TEXT,
            matched_points TEXT,
            missing_points TEXT,
            ai_score REAL,
            ai_feedback TEXT,
            teacher_score REAL,
            teacher_verified INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Uploaded',
            UNIQUE(exam_id, student_id, question_id),
            FOREIGN KEY(exam_id) REFERENCES exams(exam_id)
        )
        """
    )

    conn.commit()
    seed_default_exam(conn)
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


def save_student_answer(exam_id: str, student_id: str, question_id: str, image_paths: List[str], transcribed_text: str):
    eid = exam_id.strip().upper()
    sid = student_id.strip().upper()
    qid = question_id.strip().upper()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO submissions (exam_id, student_id, question_id, image_paths, transcribed_text, status)
        VALUES (?, ?, ?, ?, ?, 'Transcribed')
        ON CONFLICT(exam_id, student_id, question_id) DO UPDATE SET
            image_paths = excluded.image_paths,
            transcribed_text = excluded.transcribed_text,
            status = 'Transcribed',
            teacher_verified = 0
        """,
        (eid, sid, qid, json.dumps(image_paths), transcribed_text),
    )
    conn.commit()
    conn.close()


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
        cursor.execute("SELECT COUNT(DISTINCT student_id) FROM submissions WHERE exam_id=?", (eid,))
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


if __name__ == "__main__":
    init_db()
    print(f"[+] Database '{DB_NAME}' initialized.")
