import sqlite3
import json
from typing import List, Dict, Optional

DB_NAME = "grading_system.db"

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates tables for Exams, Extracted Question Paper Rubrics, and Student Submissions."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Exams Table (Stores the uploaded Question Paper metadata)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exams (
            exam_id TEXT PRIMARY KEY,
            exam_title TEXT NOT NULL,
            subject TEXT DEFAULT 'Research Methodology & IPR',
            qp_image_paths TEXT,
            raw_qp_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Rubrics Table (Stores questions extracted from the Question Paper)
    cursor.execute('''
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
    ''')
    
    # 3. Submissions Table (Stores student answers graded against that specific exam)
    cursor.execute('''
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
    ''')
    
    conn.commit()
    conn.close()

# --- EXAM & QUESTION PAPER FUNCTIONS ---

def create_exam(exam_id: str, exam_title: str, qp_image_paths: List[str], raw_qp_text: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO exams (exam_id, exam_title, qp_image_paths, raw_qp_text)
        VALUES (?, ?, ?, ?)
    ''', (exam_id.upper(), exam_title, json.dumps(qp_image_paths), raw_qp_text))
    conn.commit()
    conn.close()

def get_all_exams() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM exams ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_question_rubric(exam_id: str, question_id: str, question_text: str, max_marks: int, key_points: List[str]):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO rubrics (exam_id, question_id, question_text, max_marks, key_points)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(exam_id, question_id) DO UPDATE SET
            question_text = excluded.question_text,
            max_marks = excluded.max_marks,
            key_points = excluded.key_points
    ''', (exam_id.upper(), question_id.upper(), question_text, max_marks, json.dumps(key_points)))
    conn.commit()
    conn.close()

def get_exam_rubrics(exam_id: str) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rubrics WHERE exam_id = ? ORDER BY question_id", (exam_id.upper(),))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        item = dict(r)
        item["key_points"] = json.loads(item["key_points"])
        result.append(item)
    return result

# --- STUDENT BOOKLET FUNCTIONS ---

def save_student_answer(exam_id: str, student_id: str, question_id: str, image_paths: List[str], transcribed_text: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO submissions (exam_id, student_id, question_id, image_paths, transcribed_text, status)
        VALUES (?, ?, ?, ?, ?, 'Transcribed')
        ON CONFLICT(exam_id, student_id, question_id) DO UPDATE SET
            image_paths = excluded.image_paths,
            transcribed_text = excluded.transcribed_text,
            status = 'Transcribed'
    ''', (exam_id.upper(), student_id, question_id.upper(), json.dumps(image_paths), transcribed_text))
    conn.commit()
    conn.close()

def update_evaluation(
    exam_id: str,
    student_id: str,
    question_id: str,
    matched_points: List[str],
    missing_points: List[str],
    ai_score: float,
    ai_feedback: str
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE submissions
        SET matched_points = ?,
            missing_points = ?,
            ai_score = ?,
            ai_feedback = ?,
            teacher_score = ?,
            status = 'Evaluated'
        WHERE exam_id = ? AND student_id = ? AND question_id = ?
    ''', (
        json.dumps(matched_points),
        json.dumps(missing_points),
        ai_score,
        ai_feedback,
        ai_score,
        exam_id.upper(),
        student_id,
        question_id.upper()
    ))
    conn.commit()
    conn.close()

def verify_submission(exam_id: str, student_id: str, question_id: str, teacher_score: float, ai_feedback: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE submissions
        SET teacher_score = ?,
            ai_feedback = ?,
            teacher_verified = 1,
            status = 'Verified'
        WHERE exam_id = ? AND student_id = ? AND question_id = ?
    ''', (teacher_score, ai_feedback, exam_id.upper(), student_id, question_id.upper()))
    conn.commit()
    conn.close()

def get_exam_submissions(exam_id: str) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM submissions WHERE exam_id = ? ORDER BY student_id, question_id", (exam_id.upper(),))
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
    learning_rate: float = 0.25
):
    """
    Continuous Learning Step:
    1. Dynamically updates rubric point weights based on teacher score error (Delta).
    2. Appends teacher-verified example to 'training_dataset.jsonl' for QLoRA fine-tuning.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Fetch the submission and rubric
    cursor.execute(
        "SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND question_id=?",
        (exam_id.upper(), student_id, question_id.upper())
    )
    sub = cursor.fetchone()
    
    cursor.execute(
        "SELECT * FROM rubrics WHERE exam_id=? AND question_id=?",
        (exam_id.upper(), question_id.upper())
    )
    rub = cursor.fetchone()
    
    if not sub or not rub:
        conn.close()
        return

    matched = json.loads(sub["matched_points"]) if sub["matched_points"] else []
    key_points = json.loads(rub["key_points"])
    max_marks = float(rub["max_marks"])
    ai_score = float(sub["ai_score"] or 0.0)
    
    # Convert string key_points to weighted dicts if not already formatted
    if key_points and isinstance(key_points[0], str):
        default_w = round(max_marks / len(key_points), 2)
        key_points = [{"point": pt, "weight": default_w} for pt in key_points]

    # 1. Online Perceptron / Gradient Weight Update
    error = teacher_score - ai_score
    if abs(error) > 0.01 and matched and key_points:
        adjustment = (learning_rate * error) / len(matched)
        for item in key_points:
            if item["point"] in matched:
                item["weight"] = round(max(0.25, min(max_marks, item["weight"] + adjustment)), 3)
        
        # Save updated weights back to the rubric table
        cursor.execute(
            "UPDATE rubrics SET key_points=? WHERE exam_id=? AND question_id=?",
            (json.dumps(key_points), exam_id.upper(), question_id.upper())
        )
        conn.commit()

    # 2. Fetch up to 2 Gold-Standard teacher-verified examples for Few-Shot prompting
    conn.close()
    
    # 3. Log to JSONL for local QLoRA fine-tuning
    training_entry = {
        "exam_id": exam_id.upper(),
        "question_id": question_id.upper(),
        "question_text": rub["question_text"],
        "student_answer": sub["transcribed_text"],
        "matched_points": matched,
        "teacher_verified_score": teacher_score,
        "feedback": sub["ai_feedback"]
    }
    with open("training_dataset.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(training_entry) + "\n")

def get_verified_examples(exam_id: str, question_id: str, limit: int = 2) -> List[Dict]:
    """Fetches past teacher-approved submissions to use as Few-Shot calibration examples."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT transcribed_text, teacher_score, ai_feedback 
        FROM submissions 
        WHERE exam_id=? AND question_id=? AND teacher_verified=1
        ORDER BY submission_id DESC LIMIT ?
    ''', (exam_id.upper(), question_id.upper(), limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    # Delete old db file if schema changed, or initialize fresh
    init_db()
    print(f"[+] Exam-Centric Database '{DB_NAME}' initialized successfully.")