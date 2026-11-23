"""Automated test suite for hybrid chunking, OCR sanitization, SDMCET IA choice rules, and EduEval deterministic grading."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunker import calculate_ia_total, chunk_student_booklet, normalize_margin_markers, parse_teacher_questions
from engine import (
    generate_atomic_rubric,
    sanitize_ocr_text,
    _strip_coordinates_and_noise,
    _preprocess_page_image,
    LocalGradingEngine,
    extract_left_margin_marker,
)
from scorer import evaluate_with_edueval, clean_ocr_text
from database import (
    init_db,
    get_connection,
    get_all_exams,
    get_exam_rubrics,
    get_exam_submissions,
    get_kpi_counts,
    update_rubric_weights_from_feedback,
    save_student_answer,
    save_raw_booklet,
    get_raw_booklet,
    get_exam_booklets,
    assign_segment_to_question,
    split_at_cursor,
    delete_student_submission,
    delete_all_student_submissions,
    merge_and_reassign_submissions,
    split_submission,
    TRAINING_LOG,
)

RUBRICS = [
    {
        "question_id": "Q1A",
        "question_text": "Define research and explain its major objectives",
        "max_marks": 5,
    },
    {
        "question_id": "Q1B",
        "question_text": "Describe the systematic procedure involved in reviewing the literature",
        "max_marks": 5,
    },
    {
        "question_id": "Q2A",
        "question_text": "Classify and explain the different types of research with examples",
        "max_marks": 5,
    },
    {
        "question_id": "Q2B",
        "question_text": "Discuss the essential features of a good research design",
        "max_marks": 5,
    },
    {
        "question_id": "Q3A",
        "question_text": "Explain the various techniques used for defining a research problem",
        "max_marks": 5,
    },
    {
        "question_id": "Q3B",
        "question_text": "Discuss the important concepts associated with research design",
        "max_marks": 5,
    },
]


def assert_eq(actual, expected, msg):
    if actual != expected:
        raise AssertionError(f"{msg}\n  expected: {expected}\n  actual:   {actual}")


def test_margin_normalization():
    sample = (
        "header\n"
        "2] (6) Literature\n"
        "36) Sampling\n"
        "3] a Design\n"
        "Ans 1a: Research\n"
        "1 (a) Introduction\n"
        "3b) Hypotheses\n"
    )
    out = normalize_margin_markers(sample)
    assert "Q2B)" in out, f"Expected Q2B in: {out}"
    assert "Q3B)" in out, f"Expected Q3B in: {out}"
    assert "Q3A)" in out, f"Expected Q3A in: {out}"
    assert "Q1A)" in out, f"Expected Q1A in: {out}"
    print("  [ok] margin normalization")


def test_hybrid_chunker_does_not_collapse_to_q1a():
    booklet = """
Internal Assessment - I

Research and its Major Objectives
Research is a systematic investigation undertaken to find solutions to problems.
Its objectives include exploratory and descriptive analysis.

2] (6) The literature review is done to fill the gap and increase relevancy of the study.
Students must identify sources and write a critical summary of past publications.

3] a Concepts Associated with Research Design
Research design is the blueprint of the study covering sampling, variables, and tools.
Independent variables are manipulated while dependent variables are measured.

36) Sampling methods include random, stratified and cluster sampling.
Testing adequacy confirms whether variables are representative.
"""
    chunks = chunk_student_booklet(booklet, RUBRICS)
    keys = set(chunks.keys())
    assert "Q1A" in keys, f"Missing Q1A: {keys}"
    assert "Q2B" in keys, f"Missing Q2B: {keys}"
    assert "Q3A" in keys, f"Missing Q3A: {keys}"
    assert "Q3B" in keys, f"Missing Q3B: {keys}"
    assert keys != {"Q1A"}, f"Collapsed into Q1A only: {keys}"
    for k in keys:
        assert k in {"Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B"}, f"Illegal naked key: {k}"
    assert "systematic investigation" in chunks["Q1A"].lower()
    assert "literature" in chunks["Q2B"].lower()
    assert "blueprint" in chunks["Q3A"].lower() or "research design" in chunks["Q3A"].lower()
    assert "sampling" in chunks["Q3B"].lower()
    print(f"  [ok] hybrid chunker extracted {sorted(keys)} with zero naked keys")


def test_orphan_b_inherits_q1():
    booklet = """
Ans 1a: Research is systematic enquiry with objectives of exploration and description.

b) Literature review follows identification of sources, reading and writing the review.
"""
    chunks = chunk_student_booklet(booklet, RUBRICS)
    keys = set(chunks.keys())
    assert "Q1A" in keys and "Q1B" in keys, keys
    for k in keys:
        assert k in {"Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B"}, f"Illegal key: {k}"
    print(f"  [ok] orphan b) inherited Q1 -> {sorted(keys)}")


def test_multipage_continuation_chunking():
    booklet = """### Page 1
1(a) Research and its Major Objectives
Research is defined as systematic investigation to discover new knowledge.
Its major objectives include:
1. Exploratory research
2. Descriptive analysis

### Page 2
(Contd...)
3. Diagnostic analysis
4. Gaining basic domain knowledge
5. Evidence-based decision making for societal benefit.

### Page 3
b) Procedure for Literature Review
The literature review follows five clear stages:
1. Identify keywords and sources
2. Retrieve literature from journals

### Page 4
Q1A contd:
Additional ethical consideration: obtaining participant consent and maintaining confidentiality.
"""
    chunks = chunk_student_booklet(booklet, RUBRICS)
    keys = set(chunks.keys())
    assert "Q1A" in keys and "Q1B" in keys, keys
    assert "diagnostic analysis" in chunks["Q1A"].lower()
    assert "ethical consideration" in chunks["Q1A"].lower()
    assert "literature review" in chunks["Q1B"].lower()
    print("  [ok] multi-page continuation chunking")


def test_multi_tier_margin_segmentation():
    """Verify multi-tier margins: Q1] on line 1, a] on line 2, OCR '6' as 'b'."""
    booklet = """
1]
a] Research is defined as a systematic process to identify problems and develop solutions.
Its key objectives are exploratory and descriptive.

(6) Literature review is conducted to bridge gaps and establish academic relevance.
It synthesizes previous findings.

3]
a] Concepts associated with research design include variables, hypotheses, and sampling frames.
A proper blueprint ensures validity.

b] Sampling methods include probability and non-probability techniques.
"""
    chunks = chunk_student_booklet(booklet, RUBRICS)
    keys = set(chunks.keys())
    assert "Q1A" in keys, f"Missing Q1A: {keys}"
    assert "Q1B" in keys, f"Missing Q1B: {keys}"
    assert "Q3A" in keys, f"Missing Q3A: {keys}"
    assert "Q3B" in keys, f"Missing Q3B: {keys}"
    for k in keys:
        assert k in {"Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B"}, f"Illegal key: {k}"

    assert "systematic process" in chunks["Q1A"].lower()
    assert "literature review" in chunks["Q1B"].lower()
    assert "concepts associated" in chunks["Q3A"].lower() or "blueprint" in chunks["Q3A"].lower()
    assert "sampling methods" in chunks["Q3B"].lower()

    # Also verify single-line multi-tier marker like Q1] a]
    booklet_single_line = """
Q1] a] Research is a structured enquiry.
Q1] (6) Literature review summarizes past literature.
3] a] Design concepts include variables.
3] 6) Sampling uses stratified sampling.
"""
    chunks_sl = chunk_student_booklet(booklet_single_line, RUBRICS)
    keys_sl = set(chunks_sl.keys())
    assert "Q1A" in keys_sl and "Q1B" in keys_sl and "Q3A" in keys_sl and "Q3B" in keys_sl, keys_sl
    print("  [ok] multi-tier margin segmentation (split across lines and combined)")


def test_booklet_chunking_test_pdf_and_test1_pdf():
    """Test realistic chunking for test.pdf pattern (Q2A, Q2B, Q3A, Q3B) and test1.pdf (Q1A, Q1B, Q3A, Q3B)."""
    test_pdf_transcription = """### Page 1
2] (a) Classify and explain the different types of research with examples:
Descriptive research describes the state of affairs as it exists at present.
Analytical research uses facts or information already available.
Applied research aims at finding a solution for an immediate problem facing a society.
Fundamental research is mainly concerned with generalizations and with the formulation of a theory.

### Page 2
2] (6) Essential features of a good research design:
A good design is often characterized by flexible, appropriate, efficient, economical features.
It minimizes bias and maximizes the reliability of the data collected and analyzed.

### Page 3
3] a TECHNIQUES USED FOR DEFINING RESEARCH PROBLEM:
The research problem should be stated in a broad general way.
Next, understand the origin and nature of the problem clearly.
Survey the available literature and rephrase the research problem into working terms.

### Page 4
3b) CONCEPTS ASSOCIATED WITH RESEARCH DESIGN:
Dependent and independent variables: A concept which can take on different values is called a variable.
Extraneous variable: Independent variables that are not related to the purpose of the study.
Control: Minimizing the influence of extraneous variables.
Hypothesis: Tentative assumption made to draw out and test logical consequences.
"""
    chunks_test = chunk_student_booklet(test_pdf_transcription, RUBRICS)
    keys_test = set(chunks_test.keys())
    assert keys_test == {"Q2A", "Q2B", "Q3A", "Q3B"}, f"Expected Q2A, Q2B, Q3A, Q3B; got {keys_test}"
    for k in keys_test:
        assert k in {"Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B"}, f"Illegal key in test.pdf: {k}"

    test1_pdf_transcription = """### Page 1
Q1]
a]
Research is defined as the systematic and sequential process of defining, analyzing, & interpreting the given problem.
The major objectives of research include:
1. To define the problem statement
2. To gain basic knowledge and theories
3. To find research gaps
4. To analyze depth in a domain

### Page 2
SYSTEMATIC PROCEDURE OF REVIEWING LITERATURE
The procedure of literature review involves:
1. Identifying keywords and sources
2. Searching journals, conference proceedings, and patents
3. Evaluating and selecting literature
4. Identifying gaps and summarizing findings

### Page 3
3]
a]
TECHNIQUES USED FOR DEFINING RESEARCH PROBLEM
Techniques include:
1. Statement of problem in a general way
2. Understanding the nature of the problem
3. Surveying the available literature
4. Developing working hypotheses

### Page 4
b]
CONCEPTS ASSOCIATED WITH RESEARCH DESIGN
Key concepts:
- Variables (Dependent and Independent variables)
- Extraneous variables and confounding factors
- Research hypotheses and experimental control
- Treatment and experimental groups
"""
    chunks_test1 = chunk_student_booklet(test1_pdf_transcription, RUBRICS)
    keys_test1 = set(chunks_test1.keys())
    assert keys_test1 == {"Q1A", "Q1B", "Q3A", "Q3B"}, f"Expected Q1A, Q1B, Q3A, Q3B; got {keys_test1}"
    for k in keys_test1:
        assert k in {"Q1A", "Q1B", "Q2A", "Q2B", "Q3A", "Q3B"}, f"Illegal key in test1.pdf: {k}"
    print("  [ok] realistic booklet chunking on test.pdf & test1.pdf with zero naked Q1/Q2 keys")


def test_roman_numerals_and_internal_lists_no_split():
    """Verify that Roman numerals (i)-(x) and indented sub-lists inside an answer do NOT trigger chunk splits."""
    booklet = """
2] a) Types of research are classified as follows:
(i) Fundamental research: Basic pure science and general theories.
(ii) Applied research: Solving immediate practical problems facing society.
(iii) Qualitative research: Exploring human behavior and qualities.
(iv) Quantitative research: Numerical measurement and statistics.

   1. Descriptive studies
   2. Diagnostic studies
These sub-points must remain strictly inside Q2A without fragmenting.
"""
    chunks = chunk_student_booklet(booklet, RUBRICS)
    assert set(chunks.keys()) == {"Q2A"}, f"Expected only Q2A, got: {chunks.keys()}"
    ans = chunks["Q2A"]
    assert "(i) Fundamental" in ans
    assert "(iv) Quantitative" in ans
    assert "Descriptive studies" in ans
    print("  [ok] Roman numerals (i)-(iv) and indented sub-lists do not split answer blocks")


def test_orphan_subquestion_semantic_routing():
    """Verify semantic routing: orphan a) matches highest cosine similarity against candidate questions."""
    booklet_types = """
a) The different types of research with examples include fundamental and applied research, descriptive and analytical methods.
"""
    chunks_q2a = chunk_student_booklet(booklet_types, RUBRICS)
    assert "Q2A" in chunks_q2a, f"Expected Q2A for research types, got {chunks_q2a.keys()}"

    booklet_prob = """
a) Techniques used for defining a research problem include broad concept exploration, reviewing existing literature, and setting objectives.
"""
    chunks_q3a = chunk_student_booklet(booklet_prob, RUBRICS)
    assert "Q3A" in chunks_q3a, f"Expected Q3A for problem definition, got {chunks_q3a.keys()}"
    print("  [ok] Semantic topic routing accurately maps orphan a) to Q2A and Q3A via embeddings")


def test_raw_booklet_persistence():
    """Verify raw booklets persistence and retrieval."""
    init_db()
    eid = "IA1_RM_2026"
    sid = "TEST_RAW_001"
    raw_text = "--- Page 1 ---\nHandwritten page 1 text.\n\n--- Page 2 ---\nHandwritten page 2 text."
    images = ["page_1.png", "page_2.png"]

    save_raw_booklet(eid, sid, images, raw_text)
    booklet = get_raw_booklet(eid, sid)
    assert booklet is not None
    assert booklet["student_id"] == sid
    assert booklet["full_transcribed_text"] == raw_text
    assert booklet["image_paths"] == images

    booklets = get_exam_booklets(eid)
    student_ids = [b["student_id"] for b in booklets]
    assert sid in student_ids
    print("  [ok] raw booklet persistence and retrieval")


def test_assign_segment_with_index_0_and_all_questions():
    """Verify assigning segment explicitly starting at index 0 (handling student skipping Q1)."""
    init_db()
    eid = "IA1_RM_2026"
    sid = "TEST_STUDENT_INDEX0"
    raw_text = (
        "Classify and explain the different types of research with examples:\n"
        "Fundamental research is pure theoretical exploration.\n"
        "Applied research finds immediate practical solutions.\n\n"
        "Essential features of a good research design:\n"
        "Flexible, efficient, and economical.\n"
        "Controls extraneous variables.\n\n"
        "Techniques for defining a research problem:\n"
        "Statement in a general way, literature survey, setting objectives.\n\n"
        "Concepts associated with research design:\n"
        "Independent and dependent variables, extraneous factors, hypotheses."
    )
    save_raw_booklet(eid, sid, ["page_1.png"], raw_text)

    # 1. Assign Q2A starting at character index 0 (essential test for non-falsy start_char == 0)
    split_q2b = raw_text.find("Essential features")
    res_q2a = assign_segment_to_question(
        exam_id=eid,
        student_id=sid,
        question_id="Q2A",
        start_char=0,
        end_char=split_q2b,
    )
    subs_1 = res_q2a["submissions"]
    assert any(s["question_id"] == "Q2A" for s in subs_1)
    q2a_card = [s for s in subs_1 if s["question_id"] == "Q2A"][0]
    assert "fundamental research" in q2a_card["transcribed_text"].lower()
    assert q2a_card["ai_score"] > 0

    # 2. Assign Q2B, Q3A, Q3B
    split_q3a = raw_text.find("Techniques for defining")
    split_q3b = raw_text.find("Concepts associated")

    assign_segment_to_question(eid, sid, "Q2B", start_char=split_q2b, end_char=split_q3a)
    assign_segment_to_question(eid, sid, "Q3A", start_char=split_q3a, end_char=split_q3b)
    res_final = assign_segment_to_question(eid, sid, "Q3B", start_char=split_q3b, end_char=len(raw_text))

    final_subs = res_final["submissions"]
    qids = {s["question_id"] for s in final_subs}
    assert {"Q2A", "Q2B", "Q3A", "Q3B"}.issubset(qids)

    # Verify SDMCET IA total (max(Q1=0, Q2) + Q3)
    ia = res_final["ia_breakdown"]
    assert ia["chosen_pair"] == "Q2"
    assert ia["q2_total"] > 0
    assert ia["q3_total"] > 0
    assert ia["ia_total"] == round(ia["q2_total"] + ia["q3_total"], 2)
    print("  [ok] assign_segment starting at index 0 and live SDMCET IA calculation")


def test_split_at_cursor_index_0_and_positive():
    """Verify split_at_cursor with index 0 (reassignment) and index > 0 (two-way split)."""
    init_db()
    eid = "IA1_RM_2026"
    sid = "TEST_STUDENT_SPLIT_CURSOR"

    # Setup a Q1A submission that was mistakenly assigned
    text_q1a = (
        "First half: Dependent and independent variables are crucial in research design.\n"
        "Second half: Extraneous variables must be controlled with proper hypotheses."
    )
    save_student_answer(eid, sid, "Q1A", ["page_1.png"], text_q1a)

    # 1. Test cursor_index == 0: Full reassignment from Q1A to Q3B
    res_reassign = split_at_cursor(
        exam_id=eid,
        student_id=sid,
        source_question_id="Q1A",
        cursor_index=0,
        target_question_id="Q3B",
    )
    subs = res_reassign["submissions"]
    qids = {s["question_id"] for s in subs}
    assert "Q1A" not in qids, f"Q1A was not deleted: {qids}"
    assert "Q3B" in qids, f"Q3B missing: {qids}"
    q3b_card = [s for s in subs if s["question_id"] == "Q3B"][0]
    assert "dependent and independent" in q3b_card["transcribed_text"].lower()

    # 2. Test cursor_index > 0: Two-way split of Q3B into Q3A and Q3B
    split_pt = q3b_card["transcribed_text"].find("Second half:")
    res_split = split_at_cursor(
        exam_id=eid,
        student_id=sid,
        source_question_id="Q3B",
        cursor_index=split_pt,
        target_question_id="Q3A",
    )
    subs_split = res_split["submissions"]
    qids_split = {s["question_id"] for s in subs_split}
    assert "Q3A" in qids_split and "Q3B" in qids_split
    print("  [ok] split_at_cursor at index 0 (reassignment) and index > 0 (two-way split)")


def test_delete_student_submission():
    """Verify deleting an assigned question submission."""
    init_db()
    eid = "IA1_RM_2026"
    sid = "TEST_STUDENT_DELETE"
    save_student_answer(eid, sid, "Q1A", ["p.png"], "Research definition.")
    save_student_answer(eid, sid, "Q1B", ["p.png"], "Literature review.")

    subs = get_exam_submissions(eid, sid)
    assert len(subs) == 2

    res_del = delete_student_submission(eid, sid, "Q1A")
    subs_after = res_del["submissions"]
    qids = {s["question_id"] for s in subs_after}
    assert "Q1A" not in qids
    assert "Q1B" in qids
    print("  [ok] delete_student_submission cleanly deletes and updates IA breakdown")


def test_merge_and_reassign_and_split():
    """Verify backend merge_and_reassign_submissions and split_submission atomic operations."""
    init_db()
    eid = "IA1_RM_2026"
    sid = "TEST_STUDENT_001"

    conn = get_connection()
    conn.cursor().execute("DELETE FROM submissions WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()

    save_student_answer(eid, sid, "Q1A", ["dummy1.png"], "First part of the answer describing types of research.")
    save_student_answer(eid, sid, "Q1B", ["dummy2.png"], "Second part mentioning fundamental and applied research.")
    save_student_answer(eid, sid, "Q2A", ["dummy3.png"], "Third part covering descriptive and analytical examples.")

    subs = get_exam_submissions(eid, sid)
    qids = {s["question_id"] for s in subs}
    assert {"Q1A", "Q1B", "Q2A"}.issubset(qids)

    # 1. Test Merge & Re-Assign
    merge_result = merge_and_reassign_submissions(
        exam_id=eid,
        student_id=sid,
        source_question_ids=["Q1A", "Q1B", "Q2A"],
        target_question_id="Q2A",
    )
    merged_subs = merge_result["submissions"]
    merged_qids = {s["question_id"] for s in merged_subs}
    assert "Q1A" not in merged_qids, f"Q1A was not deleted: {merged_qids}"
    assert "Q1B" not in merged_qids, f"Q1B was not deleted: {merged_qids}"
    assert "Q2A" in merged_qids, f"Q2A missing in {merged_qids}"

    q2a_sub = [s for s in merged_subs if s["question_id"] == "Q2A"][0]
    assert "First part" in q2a_sub["transcribed_text"]
    assert "Second part" in q2a_sub["transcribed_text"]
    assert "Third part" in q2a_sub["transcribed_text"]
    assert q2a_sub["ai_score"] is not None
    assert merge_result["ia_breakdown"]["ia_total"] > 0

    # 2. Test Split Submission
    split_index = q2a_sub["transcribed_text"].find("Third part")
    assert split_index > 0
    split_result = split_submission(
        exam_id=eid,
        student_id=sid,
        source_question_id="Q2A",
        split_index=split_index,
        new_question_id="Q2B",
    )
    split_subs = split_result["submissions"]
    split_qids = {s["question_id"] for s in split_subs}
    assert "Q2A" in split_qids
    assert "Q2B" in split_qids

    q2a_after = [s for s in split_subs if s["question_id"] == "Q2A"][0]
    q2b_after = [s for s in split_subs if s["question_id"] == "Q2B"][0]
    assert "First part" in q2a_after["transcribed_text"]
    assert "Third part" not in q2a_after["transcribed_text"]
    assert "Third part" in q2b_after["transcribed_text"]
    print("  [ok] merge_and_reassign_submissions and split_submission atomic operations & IA total re-calc")


def test_generation_kwargs_clean_sampling():
    temperature = 0.0
    top_p = None
    max_tokens = 768
    repetition_penalty = 1.05

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "repetition_penalty": repetition_penalty,
    }
    if temperature and temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    assert gen_kwargs["do_sample"] is False
    assert "temperature" not in gen_kwargs
    assert "top_p" not in gen_kwargs
    assert "top_k" not in gen_kwargs
    print("  [ok] generation kwargs clean construction (no sampling flags on greedy decoding)")


def test_sanitize_hallucinations_and_coordinate_stripping():
    # Loop collapse
    looped = ("To fill the gap and increase relevancy " * 30).strip()
    cleaned = sanitize_ocr_text(looped)
    assert cleaned.lower().count("to fill the gap and increase relevancy") <= 3, cleaned

    # Synthetic follow-up quiz truncation
    quiz = (
        "Student wrote about sampling.\n"
        "Question 1: What is an experiment? Define variables.\n"
        "Question 2: What is research design?\n"
    )
    cleaned_quiz = sanitize_ocr_text(quiz)
    assert "Question 1: What is" not in cleaned_quiz
    assert "Question 2" not in cleaned_quiz

    # Visual grounding coordinates stripping
    grounded = (
        "(25,47),(998,950)\n"
        "<box>(25,47),(998,950)</box>\n"
        "<|box_start|>(25,47),(998,950)<|box_end|>\n"
        "Research is a systematic study to find solutions.\n"
        "[25, 47, 998, 950]\n"
        "(0,0),(1000,1000)"
    )
    cleaned_grounded = sanitize_ocr_text(grounded)
    assert "(25,47),(998,950)" not in cleaned_grounded
    assert "<box>" not in cleaned_grounded
    assert "<|box_start|>" not in cleaned_grounded
    assert "[25, 47, 998, 950]" not in cleaned_grounded
    assert "Research is a systematic study" in cleaned_grounded

    stripped_direct = _strip_coordinates_and_noise("(25,47),(998,950)\nValid handwriting\n(10,20)")
    assert "(25,47)" not in stripped_direct
    assert "Valid handwriting" in stripped_direct

    # Verify preserving student sub-points like (i), (ii), 2 (6), and 36) Concepts with Research design:
    subpoints_text = (
        "(25,47),(998,950)\n"
        "(i) Introduction to research methodologies.\n"
        "(ii) Systematic empirical observation.\n"
        "2 (6) Literature review protocols.\n"
        "36) Concepts with Research design:\n"
        "(100, 200), (300, 400)\n"
        "Independent and dependent variables."
    )
    preserved = _strip_coordinates_and_noise(subpoints_text)
    assert "(25,47)" not in preserved
    assert "(998,950)" not in preserved
    assert "(100, 200)" not in preserved
    assert "(i) Introduction" in preserved
    assert "(ii) Systematic" in preserved
    assert "2 (6) Literature" in preserved
    assert "36) Concepts with Research design:" in preserved
    assert "Independent and dependent variables." in preserved
    print("  [ok] OCR sanitizer & coordinate stripping ((25,47),(998,950) removed while preserving (i), (ii), 2 (6), 36))")


def test_strip_chat_prefixes_and_runaway_numbers():
    """Verify _strip_coordinates_and_noise removes chat prefixes, markdown wrappers, and runaway numbers while preserving handwritten text and short numbers."""
    # 1. Leading conversational filler and ```markdown wrappers
    filler_input = (
        "Certainly! Here is the transcription of the handwritten answer sheet:\n"
        "```markdown\n"
        "2] a) Research is defined as a systematic process to identify problems.\n"
        "```"
    )
    cleaned_filler = _strip_coordinates_and_noise(filler_input)
    assert cleaned_filler == "2] a) Research is defined as a systematic process to identify problems.", f"Got: {cleaned_filler}"

    # 2. Runaway sequential number sequence (e.g. 1 2 3 ... 12)
    runaway_input = (
        "1 2 3 4 5 6 7 8 9 10 11 12\n"
        "2] a) Concepts with research design:\n"
        "36) Sampling methods include random and stratified sampling."
    )
    cleaned_runaway = _strip_coordinates_and_noise(runaway_input)
    assert "1 2 3 4 5 6 7 8 9 10 11 12" not in cleaned_runaway
    assert "2] a) Concepts with research design:" in cleaned_runaway
    assert "36) Sampling methods include random and stratified sampling." in cleaned_runaway

    # 3. Combined test with filler, wrappers, runaway numbers, short numbers, and years
    combined_input = (
        "Certainly! Here is the transcription...\n"
        "```markdown\n"
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n"
        "2] a) Classify and explain the different types of research with examples:\n"
        "In 2024, step 1 and step 2 were evaluated.\n"
        "36) Concepts associated with research design.\n"
        "```"
    )
    cleaned_combined = _strip_coordinates_and_noise(combined_input)
    assert "Certainly!" not in cleaned_combined
    assert "```" not in cleaned_combined
    assert "1 2 3 4" not in cleaned_combined
    assert "2] a) Classify and explain the different types of research with examples:" in cleaned_combined
    assert "In 2024, step 1 and step 2 were evaluated." in cleaned_combined
    assert "36) Concepts associated with research design." in cleaned_combined
    print("  [ok] _strip_coordinates_and_noise removes chat prefixes, markdown fences, and runaway numbers while preserving 2] a), 36), and content")


def test_preprocess_page_image_crops_top_margin():
    """Verify _preprocess_page_image crops out the top ~12% height (crop_box[1] >= int(h * 0.10))."""
    from PIL import Image
    import tempfile

    w, h = 800, 1200
    img = Image.new("RGB", (w, h), color="white")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dummy_img_path = f.name
    img.save(dummy_img_path)

    try:
        preprocessed_path = _preprocess_page_image(dummy_img_path)
        assert preprocessed_path != dummy_img_path, "Expected preprocessed temp image path"
        assert os.path.exists(preprocessed_path), "Preprocessed image file does not exist"

        with Image.open(preprocessed_path) as res_img:
            w_res, h_res = res_img.size
            top_cropped = h - h_res
            assert w_res == w, f"Expected width {w}, got {w_res}"
            assert top_cropped >= int(h * 0.10), f"Expected top crop >= {int(h * 0.10)}, got {top_cropped}"
            assert top_cropped == int(h * 0.12), f"Expected top crop == {int(h * 0.12)}, got {top_cropped}"
            print(f"  [ok] _preprocess_page_image cropped {top_cropped}px top margin (>= 10% of {h}px)")
    finally:
        if os.path.exists(dummy_img_path):
            os.remove(dummy_img_path)
        if "preprocessed_path" in locals() and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


def test_choice_rule():
    # Case 1: Q1 > Q2
    scores = {"Q1A": 4.0, "Q1B": 3.5, "Q2A": 2.0, "Q2B": 2.0, "Q3A": 5.0, "Q3B": 4.0}
    result = calculate_ia_total(scores)
    assert_eq(result["ia_total"], 16.5, "IA total")
    assert_eq(result["chosen_pair"], "Q1", "chosen pair")
    assert_eq(result["q3_total"], 9.0, "Q3 total")

    # Case 2: Q2 > Q1
    scores2 = {"Q1A": 1.0, "Q1B": 1.0, "Q2A": 5.0, "Q2B": 4.5, "Q3A": 3.0, "Q3B": 3.0}
    result2 = calculate_ia_total(scores2)
    assert_eq(result2["ia_total"], 15.5, "IA total prefers Q2")
    assert_eq(result2["chosen_pair"], "Q2", "chosen pair Q2")

    # Case 3: Only Q1 and Q3 attempted
    scores3 = {"Q1A": 5.0, "Q1B": 5.0, "Q3A": 4.5, "Q3B": 4.5}
    result3 = calculate_ia_total(scores3)
    assert_eq(result3["ia_total"], 19.0, "IA total Q1+Q3")
    assert_eq(result3["chosen_pair"], "Q1", "chosen pair Q1")
    print("  [ok] SDMCET IA choice rule")


def test_teacher_parse_and_rubric():
    pasted = """
Q1 (a) Define research and explain its major objectives. 5 Marks CO-1
Q1 (b) Describe the systematic procedure involved in reviewing the literature. 5 Marks CO-2
Q3 (a) Discuss the concepts associated with research design. 5 Marks CO-3
Q3 (b) Explain sampling methods used in research. 5 Marks CO-4
"""
    qs = parse_teacher_questions(pasted)
    ids = [q["question_id"] for q in qs]
    assert ids == ["Q1A", "Q1B", "Q3A", "Q3B"], ids
    rubric = generate_atomic_rubric(qs[0]["question_text"], 5)
    assert len(rubric) == 4
    total_w = sum(p["weight"] for p in rubric)
    assert abs(total_w - 5.0) < 0.05, f"Weights don't sum to 5: {total_w}"
    print("  [ok] teacher paste parse + atomic rubric")


def test_database_and_calibration():
    init_db()
    exams = get_all_exams()
    assert len(exams) >= 1, "Expected seeded exam"
    eid = exams[0]["exam_id"]
    rubs = get_exam_rubrics(eid)
    assert len(rubs) == 6, f"Expected 6 rubrics, found {len(rubs)}"

    kpis = get_kpi_counts(eid)
    assert kpis["questions"] == 6
    assert kpis["booklets"] >= 1

    # Test weight calibration feedback
    subs = get_exam_submissions(eid, "2SD23CS082")
    calib_qid = "Q1A" if any(s["question_id"] == "Q1A" for s in subs) else subs[0]["question_id"]
    updated = update_rubric_weights_from_feedback(
        exam_id=eid,
        student_id="2SD23CS082",
        question_id=calib_qid,
        teacher_score=4.5,
    )
    assert updated is not None
    assert TRAINING_LOG.exists()
    print("  [ok] database seed & live weight calibration")


def test_transcribe_images_fallback_retry_recovers_page():
    """Verify that when Attempt 1 returns < 15 words, transcribe_images triggers Attempt 2 Top/Bottom split pass and recovers the full page transcription."""
    from PIL import Image
    import tempfile

    engine = LocalGradingEngine()
    call_log = []

    def mock_generate(messages, **kwargs):
        user_content = messages[0]["content"]
        user_text = [item["text"] for item in user_content if item["type"] == "text"][0]
        call_log.append(user_text)

        # If it's Attempt 1 (full page with prompt_attempt1), return few words (simulating failure/grounding coordinates)
        if "top to bottom" in user_text:
            return "(25,47),(998,950) only two words"
        # If it's Attempt 2 (split halves), return real words
        elif "visible in this image" in user_text:
            return "Research is a systematic and sequential process of defining and analyzing problems."
        return "Fallback text"

    engine._generate_text = mock_generate

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dummy_path = f.name

    # Save a valid test image with PIL
    img = Image.new("RGB", (400, 800), color="white")
    img.save(dummy_path)

    try:
        result = engine.transcribe_images([dummy_path], auto_unload=False)
        # Check call_log: Attempt 1 was called first, then Attempt 2 (top half and bottom half)
        assert len(call_log) >= 3, f"Expected at least 3 calls (1 initial + 2 halves), got {len(call_log)}"
        assert "coordinates" not in call_log[0].lower(), "Attempt 1 prompt should NOT mention coordinates"
        assert "(25,47)" not in call_log[0], "Attempt 1 prompt should NOT mention (25,47)"
        assert "top to bottom" in call_log[0], "Attempt 1 prompt missing positive instruction"

        # Check that result contains the recovered text from Attempt 2
        assert "systematic and sequential process" in result
        assert "--- Page 1 ---" in result
        print("  [ok] 3-stage fallback: Attempt 1 <15 words automatically triggers Attempt 2 Top/Bottom split pass and recovers full page")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)


def test_edueval_deterministic_scoring_q1a():
    """Verify evaluate_with_edueval accurately grades sample student handwritten transcription."""
    question_text = "Define research and explain its major objectives"
    max_marks = 5.0
    rubric_points = [
        {"point": "Clear and accurate definition of research as a systematic process", "weight": 1.25},
        {"point": "Primary exploratory, descriptive, and diagnostic objectives", "weight": 1.25},
        {"point": "Academic and practical significance in problem solving", "weight": 1.25},
        {"point": "Methodological advancement and societal benefit", "weight": 1.25},
    ]

    student_answer = (
        "Research is defined as the systematic and sequential process of defining, analyzing, & interpreting "
        "the given question or problem.\n"
        "Re = 'Again & again'\n"
        "Search = 'Analyzing & interpreting the answer to the problem'\n\n"
        "The major objectives of research include:\n"
        "1. To define the problem statement\n"
        "2. To gain basic knowledge and theories\n"
        "3. To find research gaps and diagnostic analysis\n"
        "4. To analyze depth in a domain\n"
        "5. For academic advancement and practical significance in problem solving\n"
        "6. To help society with evidence-based decisions and societal benefit."
    )

    eval_result = evaluate_with_edueval(
        question_text=question_text,
        max_marks=max_marks,
        key_points=rubric_points,
        student_answer=student_answer,
    )

    score = eval_result["ai_score"]
    matched = eval_result["matched_points"]
    feedback = eval_result["ai_feedback"]

    assert score in {4.5, 5.0}, f"Expected 4.5 or 5.0, got {score}"
    assert len(matched) >= 3, f"Expected at least 3 matched points, got {len(matched)}: {matched}"
    assert len(feedback) > 20, f"Empty or too short feedback: {feedback}"

    engine = LocalGradingEngine()
    engine_result = engine.evaluate_answer(
        question_text=question_text,
        max_marks=max_marks,
        key_points=rubric_points,
        student_answer=student_answer,
    )
    assert engine_result["ai_score"] in {4.5, 5.0}
    assert engine_result["matched_points"] == eval_result["matched_points"]
    print(f"  [ok] EduEval deterministic grading awarded {score}/{max_marks} marks with {len(matched)} criteria matched")


def test_dump_q1a_split_char0_and_evaluate_workflow():
    """
    End-to-end integration test of the 3-step teacher workflow:
    1. Step 1 (Upload/OCR): Dump full booklet verbatim into Q1A (ai_score=0.0, status='Transcribed').
    2. Step 2 (Cursor Split at index 0): Move entire text from Q1A to Q2A (deleting empty Q1A).
       Then sequential cursor splits (> 0) into Q2B, Q3A, Q3B.
    3. Step 3 (On-Demand Evaluation): Call POST /api/submissions/evaluate-student.
       Verify all cards are graded with EduEval, scores > 0, feedback populated,
       and SDMCET IA Choice Rule (max(Q1=0, Q2) + Q3) calculated correctly.
    """
    from fastapi.testclient import TestClient
    from main import app

    init_db()
    eid = "IA1_RM_2026"
    sid = "TEST_STUDENT_3STEP"
    client = TestClient(app)

    # Clean any prior state for this test student
    delete_all_student_submissions(eid, sid)

    # 1. Step 1: Simulate OCR dump of entire booklet into Q1A
    booklet_text = (
        "Classify and explain the different types of research with examples:\n"
        "Fundamental or pure research is driven by curiosity and theoretical understanding.\n"
        "Applied research aims to find direct solutions for pressing practical problems.\n"
        "Descriptive research reports things as they exist without manipulating variables, "
        "while analytical research uses existing data for critical evaluation.\n\n"
        "Essential features of a good research design:\n"
        "A sound research design should be flexible, appropriate, efficient, and cost-effective.\n"
        "It minimizes experimental error and bias while maximizing the reliability of the collected evidence.\n\n"
        "Explain the various techniques used for defining a research problem:\n"
        "Techniques include stating the problem in a general way, understanding its origin and nature, "
        "surveying relevant literature, and formulating clear operational hypotheses.\n\n"
        "Discuss the important concepts associated with research design:\n"
        "Key concepts include independent variables, dependent variables, extraneous variables, "
        "control mechanisms, and hypothesis testing."
    )
    save_student_answer(eid, sid, "Q1A", ["test_page_1.png", "test_page_2.png"], booklet_text)
    save_raw_booklet(eid, sid, ["test_page_1.png", "test_page_2.png"], booklet_text)

    # Verify Step 1 initial state: Exactly 1 card (Q1A) with score 0.0 and status Transcribed
    subs_initial = get_exam_submissions(eid, sid)
    assert len(subs_initial) == 1, f"Expected 1 initial submission, got {len(subs_initial)}"
    assert subs_initial[0]["question_id"] == "Q1A"
    assert subs_initial[0]["ai_score"] == 0.0
    assert subs_initial[0]["status"] == "Transcribed"

    # 2. Step 2a: split_index == 0 reassigns Q1A to Q2A and deletes empty Q1A
    resp_char0 = client.post("/api/submissions/split", json={
        "exam_id": eid,
        "student_id": sid,
        "source_question_id": "Q1A",
        "target_question_id": "Q2A",
        "split_index": 0,
    })
    assert resp_char0.status_code == 200, f"split_index=0 failed: {resp_char0.text}"
    subs_char0 = resp_char0.json()["submissions"]
    qids_char0 = {s["question_id"] for s in subs_char0}
    assert "Q1A" not in qids_char0, f"Q1A was not deleted: {qids_char0}"
    assert "Q2A" in qids_char0, f"Q2A missing after split_index=0: {qids_char0}"
    assert len(subs_char0) == 1

    # 2. Step 2b: Sequential splits with split_index > 0
    # Split Q2A -> Q2B
    q2a_card = [s for s in subs_char0 if s["question_id"] == "Q2A"][0]
    split_q2b = q2a_card["transcribed_text"].find("Essential features")
    assert split_q2b > 0, "Could not find split boundary for Q2B"

    resp_q2b = client.post("/api/submissions/split", json={
        "exam_id": eid,
        "student_id": sid,
        "source_question_id": "Q2A",
        "target_question_id": "Q2B",
        "split_index": split_q2b,
    })
    assert resp_q2b.status_code == 200
    subs_q2b = resp_q2b.json()["submissions"]
    qids_q2b = {s["question_id"] for s in subs_q2b}
    assert qids_q2b == {"Q2A", "Q2B"}

    # Split Q2B -> Q3A
    q2b_card = [s for s in subs_q2b if s["question_id"] == "Q2B"][0]
    split_q3a = q2b_card["transcribed_text"].find("Explain the various techniques")
    assert split_q3a > 0, "Could not find split boundary for Q3A"

    resp_q3a = client.post("/api/submissions/split", json={
        "exam_id": eid,
        "student_id": sid,
        "source_question_id": "Q2B",
        "target_question_id": "Q3A",
        "split_index": split_q3a,
    })
    assert resp_q3a.status_code == 200
    subs_q3a = resp_q3a.json()["submissions"]
    qids_q3a = {s["question_id"] for s in subs_q3a}
    assert qids_q3a == {"Q2A", "Q2B", "Q3A"}

    # Split Q3A -> Q3B
    q3a_card = [s for s in subs_q3a if s["question_id"] == "Q3A"][0]
    split_q3b = q3a_card["transcribed_text"].find("Discuss the important concepts")
    assert split_q3b > 0, "Could not find split boundary for Q3B"

    resp_q3b = client.post("/api/submissions/split", json={
        "exam_id": eid,
        "student_id": sid,
        "source_question_id": "Q3A",
        "target_question_id": "Q3B",
        "split_index": split_q3b,
    })
    assert resp_q3b.status_code == 200
    subs_final = resp_q3b.json()["submissions"]
    qids_final = sorted([s["question_id"] for s in subs_final])
    assert qids_final == ["Q2A", "Q2B", "Q3A", "Q3B"], f"Unexpected question set: {qids_final}"

    # 3. Step 3: Trigger On-Demand Evaluation across all 4 assigned cards
    resp_eval = client.post("/api/submissions/evaluate-student", json={
        "exam_id": eid,
        "student_id": sid,
    })
    assert resp_eval.status_code == 200, f"evaluate-student failed: {resp_eval.text}"
    eval_data = resp_eval.json()
    evaluated_subs = eval_data["submissions"]
    assert len(evaluated_subs) == 4

    for sub in evaluated_subs:
        qid = sub["question_id"]
        assert sub["ai_score"] > 0, f"{qid} ai_score was not graded: {sub['ai_score']}"
        assert len(sub["matched_points"]) > 0, f"{qid} has 0 matched points"
        assert len(sub["ai_feedback"]) > 10, f"{qid} ai_feedback is empty or too short"

    # Verify SDMCET IA Choice Rule breakdown: max(Q1=0, Q2) + Q3
    ia = eval_data["ia_breakdown"]
    assert ia["chosen_pair"] == "Q2", f"Expected Q2 chosen, got {ia['chosen_pair']}"
    assert ia["q1_total"] == 0.0
    assert ia["q2_total"] > 0.0
    assert ia["q3_total"] > 0.0
    assert eval_data["ia_total"] == round(ia["q2_total"] + ia["q3_total"], 2)
    print(f"  [ok] 3-step workflow: Dump Q1A -> Char 0 Reassign -> Split Q2B/Q3A/Q3B -> EduEval IA Total: {eval_data['ia_total']}/20")


if __name__ == "__main__":
    print("Running EvalAI pipeline tests...")
    test_margin_normalization()
    test_hybrid_chunker_does_not_collapse_to_q1a()
    test_orphan_b_inherits_q1()
    test_multipage_continuation_chunking()
    test_multi_tier_margin_segmentation()
    test_booklet_chunking_test_pdf_and_test1_pdf()
    test_roman_numerals_and_internal_lists_no_split()
    test_orphan_subquestion_semantic_routing()
    test_raw_booklet_persistence()
    test_assign_segment_with_index_0_and_all_questions()
    test_split_at_cursor_index_0_and_positive()
    test_delete_student_submission()
    test_merge_and_reassign_and_split()
    test_dump_q1a_split_char0_and_evaluate_workflow()
    test_generation_kwargs_clean_sampling()
    test_sanitize_hallucinations_and_coordinate_stripping()
    test_strip_chat_prefixes_and_runaway_numbers()
    test_preprocess_page_image_crops_top_margin()
    test_choice_rule()
    test_teacher_parse_and_rubric()
    test_database_and_calibration()
    test_transcribe_images_fallback_retry_recovers_page()
    test_edueval_deterministic_scoring_q1a()
    print("\nAll EvalAI pipeline tests passed successfully!")
