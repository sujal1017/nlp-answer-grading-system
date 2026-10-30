import os
import json
import streamlit as st
from database import (
    init_db,
    create_exam,
    get_all_exams,
    save_question_rubric,
    get_exam_rubrics,
    save_student_answer,
    update_evaluation,
    verify_submission,
    update_rubric_weights_from_feedback,
    get_verified_examples,
    get_exam_submissions
)
from chunker import (
    parse_sdmcet_question_paper,
    chunk_student_booklet,
    calculate_ia_total
)
from engine import LocalGradingEngine

# Ensure database tables exist
init_db()

st.set_page_config(page_title="AI Exam Grader - SDMCET", layout="wide")
st.title("🎓 Local AI Handwritten Booklet Grader & Continuous Learner")

@st.cache_resource
def get_engine():
    """Singleton instance so Qwen2-VL is only managed once per session."""
    return LocalGradingEngine()

engine = get_engine()

# Sidebar: Exam Selection & VRAM Controls
st.sidebar.header("⚙️ Exam & GPU Controls")
exams = get_all_exams()
exam_options = [e["exam_id"] for e in exams]
selected_exam = st.sidebar.selectbox(
    "Active Exam ID",
    options=exam_options if exam_options else ["No Exams Created"]
)

if st.sidebar.button("🧹 Free GPU VRAM"):
    engine.unload_model()
    st.sidebar.success("Cleared Qwen2-VL from RTX 4050 VRAM!")

tab1, tab2, tab3 = st.tabs([
    "📄 1. Upload Question Paper & Rubrics",
    "📝 2. Upload & Grade Student Booklet",
    "✅ 3. Teacher Verification & Weight Calibration"
])

# ==============================================================================
# TAB 1: QUESTION PAPER INGESTION & RUBRIC LOCK-IN
# ==============================================================================
with tab1:
    st.subheader("Create Exam from Question Paper")
    col_a, col_b = st.columns(2)
    with col_a:
        new_exam_id = st.text_input("Exam ID (e.g., IA1_DSA_2026 or IA1_RM_2026)", value="IA1_2026").strip().upper()
    with col_b:
        new_exam_title = st.text_input("Course Title", value="Research Methodology & IPR / Distributed Systems")

    qp_files = st.file_uploader(
        "Upload Question Paper Image(s)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    manual_qp_text = st.text_area(
        "Or Paste Question Paper Text Directly (Optional - skips OCR if provided):",
        height=140,
        placeholder="Q. No. 1\na) Define a research problem... [5M, CO:1]\nb) Explain literature review... [5M, CO:1]"
    )

    if st.button("🚀 Extract Questions & Auto-Generate Weighted Rubrics"):
        if not new_exam_id:
            st.error("Please enter a valid Exam ID.")
        else:
            qp_paths = []
            raw_qp_md = manual_qp_text.strip()

            if qp_files and not raw_qp_md:
                qp_dir = os.path.join("data", "exams", new_exam_id)
                os.makedirs(qp_dir, exist_ok=True)
                for f in qp_files:
                    p = os.path.join(qp_dir, f.name)
                    with open(p, "wb") as out:
                        out.write(f.getbuffer())
                    qp_paths.append(p)

                with st.spinner("Running Qwen2-VL OCR on Question Paper..."):
                    raw_qp_md = engine.transcribe_images(qp_paths, is_question_paper=True)

            if not raw_qp_md:
                st.error("Please either upload a Question Paper image or paste the text.")
            else:
                create_exam(new_exam_id, new_exam_title, qp_paths, raw_qp_md)
                parsed_questions = parse_sdmcet_question_paper(raw_qp_md)

                if not parsed_questions:
                    st.warning("Could not detect standard 'a) / b)' sub-questions. Saving raw text as Q1A.")
                    parsed_questions = [{"question_id": "Q1A", "question_text": raw_qp_md, "max_marks": 5}]

                with st.spinner(f"Generating weighted rubrics for {len(parsed_questions)} questions..."):
                    for q in parsed_questions:
                        weighted_pts = engine.generate_rubric_points(q["question_text"], q["max_marks"])
                        save_question_rubric(
                            new_exam_id,
                            q["question_id"],
                            q["question_text"],
                            q["max_marks"],
                            weighted_pts
                        )
                engine.unload_model()
                st.success(f"Exam '{new_exam_id}' created with {len(parsed_questions)} questions! Refresh or select it in the sidebar.")
                st.rerun()

    st.divider()
    if selected_exam != "No Exams Created":
        st.subheader(f"Active Rubrics & Live Weights for `{selected_exam}`")
        rubrics = get_exam_rubrics(selected_exam)
        for r in rubrics:
            with st.expander(f"**{r['question_id']}** ({r['max_marks']} Marks): {r['question_text']}"):
                q_text_edit = st.text_input(
                    f"Question Prompt ({r['question_id']})",
                    value=r["question_text"],
                    key=f"qtxt_{r['question_id']}"
                )
                max_m_edit = st.number_input(
                    f"Max Marks ({r['question_id']})",
                    value=int(r["max_marks"]),
                    min_value=1,
                    max_value=20,
                    key=f"qm_{r['question_id']}"
                )
                pts_json_str = st.text_area(
                    f"Weighted Key Points JSON ({r['question_id']})",
                    value=json.dumps(r["key_points"], indent=2),
                    height=180,
                    key=f"qpts_{r['question_id']}"
                )
                if st.button(f"💾 Save Rubric Changes for {r['question_id']}", key=f"btn_{r['question_id']}"):
                    try:
                        updated_pts = json.loads(pts_json_str)
                        save_question_rubric(selected_exam, r["question_id"], q_text_edit, max_m_edit, updated_pts)
                        st.success(f"Updated {r['question_id']} rubric!")
                    except Exception as e:
                        st.error(f"Invalid JSON format: {e}")

# ==============================================================================
# TAB 2: STUDENT BOOKLET UPLOAD, CHUNKING & AI GRADING
# ==============================================================================
with tab2:
    st.subheader(f"Process Student Answer Booklet — Exam: `{selected_exam}`")
    if selected_exam == "No Exams Created":
        st.info("Please create an exam in Tab 1 first.")
    else:
        student_usn = st.text_input("Student USN / Roll Number (e.g., 2SD23CS082)", value="2SD23CS082").strip().upper()
        booklet_pages = st.file_uploader(
            "Upload Student Booklet Pages in Order (page_1, page_2...)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("🔍 Run Multi-Page OCR, Chunking & Rubric Evaluation"):
            if not student_usn or not booklet_pages:
                st.error("Please provide a Student USN and upload at least one answer page.")
            else:
                student_dir = os.path.join("data", "booklets", selected_exam, student_usn)
                os.makedirs(student_dir, exist_ok=True)
                saved_page_paths = []
                for idx, page_file in enumerate(booklet_pages):
                    ext = os.path.splitext(page_file.name)[1]
                    page_path = os.path.join(student_dir, f"page_{idx+1}{ext}")
                    with open(page_path, "wb") as out:
                        out.write(page_file.getbuffer())
                    saved_page_paths.append(page_path)

                with st.spinner("Step 1/3: Running Qwen2-VL OCR across booklet pages..."):
                    full_booklet_md = engine.transcribe_images(saved_page_paths, is_question_paper=False)

                rubrics = get_exam_rubrics(selected_exam)
                rubric_map = {r["question_id"].upper(): r for r in rubrics}
                expected_qids = list(rubric_map.keys())

                with st.spinner("Step 2/3: Chunking booklet into individual question answers..."):
                    chunks = chunk_student_booklet(full_booklet_md, expected_q_ids=expected_qids)

                with st.spinner("Step 3/3: Evaluating chunks with live calibrated weights & Few-Shot examples..."):
                    for q_id, ans_text in chunks.items():
                        save_student_answer(selected_exam, student_usn, q_id, saved_page_paths, ans_text)
                        if q_id in rubric_map:
                            r_info = rubric_map[q_id]
                            past_examples = get_verified_examples(selected_exam, q_id, limit=2)
                            eval_res = engine.evaluate_answer(
                                question_text=r_info["question_text"],
                                max_marks=r_info["max_marks"],
                                key_points=r_info["key_points"],
                                student_answer=ans_text,
                                verified_examples=past_examples
                            )
                            update_evaluation(
                                exam_id=selected_exam,
                                student_id=student_usn,
                                question_id=q_id,
                                matched_points=eval_res["matched_points"],
                                missing_points=eval_res["missing_points"],
                                ai_score=eval_res["ai_score"],
                                ai_feedback=eval_res["ai_feedback"]
                            )
                engine.unload_model()
                st.success(f"Processed {len(chunks)} answers for {student_usn}! Go to Tab 3 to verify and calibrate.")

# ==============================================================================
# TAB 3: TEACHER VERIFICATION & CONTINUOUS WEIGHT CALIBRATION
# ==============================================================================
with tab3:
    st.subheader(f"Teacher Verification & Continuous Learning — Exam: `{selected_exam}`")
    if selected_exam != "No Exams Created":
        submissions = get_exam_submissions(selected_exam)
        if not submissions:
            st.info("No student submissions found for this exam yet.")
        else:
            students = sorted(list({s["student_id"] for s in submissions}))
            selected_student = st.selectbox("Select Student USN", options=students)
            student_subs = [s for s in submissions if s["student_id"] == selected_student]

            # Calculate SDMCET IA Total (Best of Q1 vs Q2 + Compulsory Q3)
            scores_dict = {s["question_id"]: float(s["teacher_score"] or 0.0) for s in student_subs}
            ia_total = calculate_ia_total(scores_dict)
            st.metric(
                label=f"Calculated IA Total for {selected_student} (Max(Q1, Q2) + Q3)",
                value=f"{ia_total} / 20 Marks"
            )

            col_left, col_right = st.columns([1, 1.2])

            with col_left:
                st.markdown("### 🖼️ Scanned Answer Booklet Pages")
                sample_paths = student_subs[0]["image_paths"] if student_subs else []
                for p_idx, img_p in enumerate(sample_paths):
                    if os.path.exists(img_p):
                        st.image(img_p, caption=f"Page {p_idx + 1}", use_container_width=True)

            with col_right:
                st.markdown("### 🧠 Chunked Answers & Live Rubric Calibration")
                for sub in student_subs:
                    qid = sub["question_id"]
                    verified_badge = "✅ Verified" if sub["teacher_verified"] else "⏳ Pending Review"
                    with st.expander(f"**{qid}** | Score: {sub['teacher_score']} | {verified_badge}", expanded=True):
                        st.markdown("**OCR Transcribed Answer:**")
                        st.info(sub["transcribed_text"])

                        st.markdown(f"**Matched Rubric Points:** `{sub['matched_points']}`")
                        st.markdown(f"**Missing Rubric Points:** `{sub['missing_points']}`")
                        st.caption(f"Initial AI Score: {sub['ai_score']}")

                        new_score = st.number_input(
                            f"Teacher Final Score ({qid})",
                            min_value=0.0,
                            max_value=20.0,
                            value=float(sub["teacher_score"] or 0.0),
                            step=0.5,
                            key=f"score_{selected_student}_{qid}"
                        )
                        new_feedback = st.text_area(
                            f"Evaluation Feedback ({qid})",
                            value=sub["ai_feedback"] or "",
                            key=f"fb_{selected_student}_{qid}"
                        )

                        if st.button(f"✅ Approve & Calibrate Weights ({qid})", key=f"verify_{selected_student}_{qid}"):
                            verify_submission(selected_exam, selected_student, qid, new_score, new_feedback)
                            update_rubric_weights_from_feedback(
                                exam_id=selected_exam,
                                student_id=selected_student,
                                question_id=qid,
                                teacher_score=new_score
                            )
                            st.success(
                                f"Approved {qid}! Rubric weights adjusted from teacher delta and saved to training_dataset.jsonl."
                            )
                            st.rerun()