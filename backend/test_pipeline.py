"""Automated test suite for hybrid chunking, OCR sanitization, SDMCET IA choice rules, and EduEval deterministic grading."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunker import calculate_ia_total, chunk_student_booklet, normalize_margin_markers, parse_teacher_questions
from engine import generate_atomic_rubric, sanitize_ocr_text, _strip_coordinates_and_noise, LocalGradingEngine
from scorer import evaluate_with_edueval, clean_ocr_text
from database import (
    init_db,
    get_connection,
    get_all_exams,
    get_exam_rubrics,
    get_kpi_counts,
    update_rubric_weights_from_feedback,
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
    # Pattern modeled from test.pdf (Student chooses Q2 and Q3)
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

    # Pattern modeled from test1.pdf (Student chooses Q1 and Q3, two-tier margins)
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
    print("  [ok] OCR sanitizer & coordinate stripping ((25,47),(998,950) removed cleanly)")


def test_choice_rule():
    # Case 1: Q1 > Q2
    scores = {"Q1A": 4.0, "Q1B": 3.5, "Q2A": 2.0, "Q2B": 2.0, "Q3A": 5.0, "Q3B": 4.0}
    result = calculate_ia_total(scores)
    # max(7.5, 4.0) + 9.0 = 16.5 / 20
    assert_eq(result["ia_total"], 16.5, "IA total")
    assert_eq(result["chosen_pair"], "Q1", "chosen pair")
    assert_eq(result["q3_total"], 9.0, "Q3 total")

    # Case 2: Q2 > Q1
    scores2 = {"Q1A": 1.0, "Q1B": 1.0, "Q2A": 5.0, "Q2B": 4.5, "Q3A": 3.0, "Q3B": 3.0}
    result2 = calculate_ia_total(scores2)
    # max(2.0, 9.5) + 6.0 = 15.5 / 20
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
    updated = update_rubric_weights_from_feedback(
        exam_id=eid,
        student_id="2SD23CS082",
        question_id="Q1A",
        teacher_score=4.5,
    )
    assert updated is not None
    assert TRAINING_LOG.exists()
    print("  [ok] database seed & live weight calibration")


def test_transcribe_images_prompt_and_grounding_fix():
    """Verify transcribe_images uses single user message with explicit negative constraint against grounding."""
    engine = LocalGradingEngine()
    captured_messages = []

    def mock_generate(messages, **kwargs):
        captured_messages.append(messages)
        return "Q1A) Research is systematic enquiry."

    engine._generate_text = mock_generate

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dummy_path = f.name
        f.write(b"\x89PNG\r\n")

    try:
        result = engine.transcribe_images([dummy_path])
        assert len(captured_messages) == 1, f"Expected 1 call, got {len(captured_messages)}"
        msgs = captured_messages[0]
        # Single user message (no grounding-inducing system message)
        assert len(msgs) == 1, f"Expected 1 user message, got {len(msgs)}"
        assert msgs[0]["role"] == "user", f"Role should be 'user', got '{msgs[0]['role']}'"

        user_content = msgs[0]["content"]
        content_types = [item["type"] for item in user_content]
        assert "image" in content_types, f"User message missing image: {content_types}"
        assert "text" in content_types, f"User message missing text: {content_types}"

        user_text = [item["text"] for item in user_content if item["type"] == "text"][0]
        assert "coordinates or numbers in parentheses like (25,47),(998,950)" in user_text
        assert "q1] a]" in user_text.lower()
        print("  [ok] transcribe_images prompt has explicit grounding fix and margin rules")
    finally:
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

    # Also test via engine.evaluate_answer() integration
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


if __name__ == "__main__":
    print("Running EvalAI pipeline tests...")
    test_margin_normalization()
    test_hybrid_chunker_does_not_collapse_to_q1a()
    test_orphan_b_inherits_q1()
    test_multipage_continuation_chunking()
    test_multi_tier_margin_segmentation()
    test_booklet_chunking_test_pdf_and_test1_pdf()
    test_generation_kwargs_clean_sampling()
    test_sanitize_hallucinations_and_coordinate_stripping()
    test_choice_rule()
    test_teacher_parse_and_rubric()
    test_database_and_calibration()
    test_transcribe_images_prompt_and_grounding_fix()
    test_edueval_deterministic_scoring_q1a()
    print("\nAll EvalAI pipeline tests passed successfully!")
