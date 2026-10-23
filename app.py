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

# Initialize DB
init_db()

st.set_page_config(
    page_title="EvalAI | Automated Booklet Grading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM MODERN UI CSS (HIDES DEFAULT STREAMLIT CHROME)
# ==============================================================================
st.markdown("""
<style>
    /* Import Google Font 'Inter' */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Default Streamlit Header, Footer, Hamburger Menu & Deploy Button */
    #MainMenu {visibility: hidden !important;}
    header {visibility: hidden !important; height: 0px !important;}
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; height: 0px !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}

    /* Reduce top padding left behind by hidden header */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #FFFFFF !important;
    }

    /* Custom Top App Banner */
    .top-nav {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 14px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
        border: 1px solid #334155;
    }
    .top-nav-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .top-nav-subtitle {
        font-size: 0.875rem;
        color: #94A3B8;
        margin-top: 0.25rem;
    }
    .gpu-badge {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Metric KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.15rem 1.35rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        margin-bottom: 1rem;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }

    /* Modern Section Cards */
    .surface-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Status Pills */
    .pill-verified {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pill-pending {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pill-marks {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Modern Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        padding: 0 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #475569;
        background-color: transparent;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    }

    /* Primary Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 2px 5px rgba(79, 70, 229, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return LocalGradingEngine()

engine = get_engine()

# ==============================================================================
# SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚡ EvalAI Studio")
    st.caption("Local VLM Grading & Continuous Calibration")
    st.markdown("---")

    exams = get_all_exams()
    exam_options = [e["exam_id"] for e in exams]
    selected_exam = st.selectbox(
        "Active Assessment Workspace",
        options=exam_options if exam_options else ["No Exams Created"]
    )

    st.markdown("---")
    st.markdown("##### 🖥️ Hardware Telemetry")
    st.caption("Engine: `Qwen2-VL-2B-Instruct`")
    st.caption("Device: `NVIDIA RTX 4050 (6GB)`")

    if st.button("🧹 Flush GPU VRAM Cache", use_container_width=True):
        engine.unload_model()
        st.toast("GPU VRAM cache cleared!", icon="✅")

# ==============================================================================
# TOP HEADER BANNER & KPI METRICS
# ==============================================================================
st.markdown(f"""
<div class="top-nav">
    <div>
        <p class="top-nav-title">Automated Handwritten Answer Evaluation System</p>
        <p class="top-nav-subtitle">Offline VLM Digitization • Atomic Rubric Alignment • Teacher-in-the-Loop Calibration</p>
    </div>
    <div class="gpu-badge">
        ● CUDA Accelerated • Active Exam: {selected_exam}
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch live stats for KPI bar
active_rubrics = get_exam_rubrics(selected_exam) if selected_exam != "No Exams Created" else []
active_subs = get_exam_submissions(selected_exam) if selected_exam != "No Exams Created" else []
unique_students = len({s["student_id"] for s in active_subs})
verified_count = sum(1 for s in active_subs if s["teacher_verified"] == 1)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Active Exam ID</div>
        <div class="kpi-value">{selected_exam if selected_exam != "No Exams Created" else "—"}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Configured Questions</div>
        <div class="kpi-value">{len(active_rubrics)}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Booklets Digitized</div>
        <div class="kpi-value">{unique_students}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Teacher Verified Answers</div>
        <div class="kpi-value">{verified_count} / {len(active_subs)}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# MAIN WORKSPACE TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "01. Question Paper & Rubric Studio",
    "02. Booklet Digitization & Grading",
    "03. Teacher Review & Live Calibration"
])

# ------------------------------------------------------------------------------
# TAB 1: QUESTION PAPER INGESTION
# ------------------------------------------------------------------------------
with tab1:
    col_setup, col_rubrics = st.columns([1, 1.35], gap="large")

    with col_setup:
        st.markdown('<div class="surface-card">', unsafe_allow_html=True)
        st.markdown("#### 📄 Initialize New Exam Workspace")
        st.caption("Upload a scanned Question Paper or paste the text to generate weighted rubrics.")

        new_exam_id = st.text_input("Exam Identifier", value="IA1_2026").strip().upper()
        new_exam_title = st.text_input("Subject / Course Title", value="Research Methodology & IPR")

        qp_files = st.file_uploader(
            "Question Paper Scan (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        manual_qp_text = st.text_area(
            "Or Paste Question Paper Text Directly:",
            height=130,
            placeholder="Q. No. 1\na) Define a research problem... [5M, CO:1]\nb) Explain literature review... [5M, CO:1]"
        )

        if st.button("✨ Extract Questions & Build Rubrics", use_container_width=True):
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
                    st.error("Upload an image or paste the question paper text first.")
                else:
                    create_exam(new_exam_id, new_exam_title, qp_paths, raw_qp_md)
                    parsed_questions = parse_sdmcet_question_paper(raw_qp_md)

                    if not parsed_questions:
                        parsed_questions = [{"question_id": "Q1A", "question_text": raw_qp_md, "max_marks": 5}]

                    with st.spinner(f"Synthesizing atomic rubrics for {len(parsed_questions)} questions..."):
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
                    st.toast(f"Workspace {new_exam_id} initialized!", icon="🎉")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rubrics:
        st.markdown("#### 🎯 Active Question Rubrics & Calibrated Weights")
        if not active_rubrics:
            st.info("No rubrics loaded. Create an exam on the left to populate questions.")
        else:
            for r in active_rubrics:
                with st.expander(f"**{r['question_id']}** — {r['question_text'][:75]}... ({r['max_marks']}M)"):
                    q_text_edit = st.text_input(
                        "Question Prompt",
                        value=r["question_text"],
                        key=f"qtxt_{r['question_id']}"
                    )
                    max_m_edit = st.number_input(
                        "Maximum Marks",
                        value=int(r["max_marks"]),
                        min_value=1,
                        max_value=20,
                        key=f"qm_{r['question_id']}"
                    )
                    pts_json_str = st.text_area(
                        "Atomic Rubric Points & Weights (JSON)",
                        value=json.dumps(r["key_points"], indent=2),
                        height=150,
                        key=f"qpts_{r['question_id']}"
                    )
                    if st.button(f"Update {r['question_id']} Rubric", key=f"btn_{r['question_id']}"):
                        try:
                            updated_pts = json.loads(pts_json_str)
                            save_question_rubric(selected_exam, r["question_id"], q_text_edit, max_m_edit, updated_pts)
                            st.toast(f"Saved {r['question_id']}!", icon="💾")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Invalid JSON: {e}")

# ------------------------------------------------------------------------------
# TAB 2: STUDENT BOOKLET UPLOAD & GRADING
# ------------------------------------------------------------------------------
with tab2:
    st.markdown('<div class="surface-card">', unsafe_allow_html=True)
    st.markdown(f"#### 📝 Batch Booklet Digitization — Workspace: `{selected_exam}`")
    st.caption("Upload multi-page handwritten booklets in sequential order. Cross-page answers are merged automatically.")

    if selected_exam == "No Exams Created":
        st.warning("Create or select an exam workspace in Tab 01 first.")
    else:
        u_col1, u_col2 = st.columns([1, 2])
        with u_col1:
            student_usn = st.text_input("Student USN / Roll ID", value="2SD23CS082").strip().upper()
        with u_col2:
            booklet_pages = st.file_uploader(
                "Upload Handwritten Pages (page_1, page_2...)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True
            )

        if st.button("🚀 Run Multi-Page OCR & Automated Grading", use_container_width=True):
            if not student_usn or not booklet_pages:
                st.error("Provide a valid Student USN and upload at least one page.")
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

                with st.spinner("1/3 • Extracting handwritten text with Qwen2-VL..."):
                    full_booklet_md = engine.transcribe_images(saved_page_paths, is_question_paper=False)

                rubric_map = {r["question_id"].upper(): r for r in active_rubrics}
                expected_qids = list(rubric_map.keys())

                with st.spinner("2/3 • Segmenting booklet by question boundaries..."):
                    chunks = chunk_student_booklet(full_booklet_md, expected_q_ids=expected_qids)

                with st.spinner("3/3 • Grading chunks against calibrated rubric weights..."):
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
                st.toast(f"Evaluated {len(chunks)} answers for {student_usn}!", icon="✅")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: TEACHER REVIEW & CONTINUOUS CALIBRATION
# ------------------------------------------------------------------------------
with tab3:
    if not active_subs:
        st.info("No evaluated student booklets yet. Upload a booklet in Tab 02.")
    else:
        students = sorted(list({s["student_id"] for s in active_subs}))
        top_bar_l, top_bar_r = st.columns([1, 2])

        with top_bar_l:
            selected_student = st.selectbox("Select Student USN to Review", options=students)

        student_subs = [s for s in active_subs if s["student_id"] == selected_student]
        scores_dict = {s["question_id"]: float(s["teacher_score"] or 0.0) for s in student_subs}
        ia_total = calculate_ia_total(scores_dict)

        with top_bar_r:
            st.markdown(f"""
            <div class="surface-card" style="display:flex; justify-content:space-between; align-items:center; padding: 0.9rem 1.4rem;">
                <div>
                    <span style="font-size:0.8rem; color:#64748B; font-weight:600;">SDMCET IA CHOICE RULE TOTAL (MAX(Q1, Q2) + Q3)</span><br>
                    <span style="font-size:0.9rem; color:#0F172A;">Student USN: <b>{selected_student}</b></span>
                </div>
                <div style="font-size:1.6rem; font-weight:700; color:#4F46E5;">
                    {ia_total} / 20 Marks
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_img, col_eval = st.columns([1, 1.25], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Original Answer Booklet")
            sample_paths = student_subs[0]["image_paths"] if student_subs else []
            for p_idx, img_p in enumerate(sample_paths):
                if os.path.exists(img_p):
                    st.image(img_p, caption=f"Scanned Page {p_idx + 1}", use_container_width=True)

        with col_eval:
            st.markdown("#### 🧠 AI Grading & Teacher Override")
            for sub in student_subs:
                qid = sub["question_id"]
                badge_html = (
                    '<span class="pill-verified">✓ Teacher Verified</span>'
                    if sub["teacher_verified"]
                    else '<span class="pill-pending">● Pending Verification</span>'
                )

                st.markdown(f"""
                <div class="surface-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                        <span style="font-size:1.1rem; font-weight:700; color:#0F172A;">Question {qid}</span>
                        <div>
                            <span class="pill-marks">AI Score: {sub['ai_score']}</span>
                            {badge_html}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Inspect Transcription & Calibrate {qid}", expanded=not sub["teacher_verified"]):
                    st.markdown("**Digitized Student Answer:**")
                    st.info(sub["transcribed_text"])

                    c_match, c_miss = st.columns(2)
                    with c_match:
                        st.markdown("**✅ Matched Points**")
                        for pt in sub["matched_points"]:
                            st.caption(f"• {pt}")
                    with c_miss:
                        st.markdown("**❌ Missing Points**")
                        for pt in sub["missing_points"]:
                            st.caption(f"• {pt}")

                    new_score = st.number_input(
                        f"Final Approved Score ({qid})",
                        min_value=0.0,
                        max_value=20.0,
                        value=float(sub["teacher_score"] or 0.0),
                        step=0.5,
                        key=f"score_{selected_student}_{qid}"
                    )
                    new_feedback = st.text_area(
                        f"Examiner Remarks ({qid})",
                        value=sub["ai_feedback"] or "",
                        height=80,
                        key=f"fb_{selected_student}_{qid}"
                    )

                    if st.button(f"Approve Grade & Calibrate Rubric ({qid})", key=f"verify_{selected_student}_{qid}", use_container_width=True):
                        verify_submission(selected_exam, selected_student, qid, new_score, new_feedback)
                        update_rubric_weights_from_feedback(
                            exam_id=selected_exam,
                            student_id=selected_student,
                            question_id=qid,
                            teacher_score=new_score
                        )
                        st.toast(f"{qid} verified! Rubric weights updated for next student.", icon="🎯")
                        st.rerun()