import math

import pytest

from app.services.answer_mapping import map_answers
from app.services.metrics import compute_metrics
from app.services.question_extraction import detect_qtype, parse_question_paper

from tests.conftest import QUESTION_PAPER

LABELS = ["Q1A", "Q1B", "Q2A", "Q2B", "Q3A"]


# ----------------------------------------------------------------- question extraction
def test_extracts_sdmcet_style_paper():
    r = parse_question_paper([(1, QUESTION_PAPER)])
    qs = {q["label"]: q for q in r["questions"]}
    assert list(qs) == LABELS
    assert all(q["max_marks"] == 5 for q in qs.values()) and not any(q["needs_review"] for q in qs.values())
    assert "[5M" not in qs["Q1A"]["text"] and qs["Q1A"]["source_page"] == 1
    assert "COMPULSORY" in r["instructions"]


def test_missing_marks_are_flagged_not_invented():
    r = parse_question_paper([(1, "Q1\na) Explain paging.\nb) Explain segmentation. [4 marks]")])
    qs = {q["label"]: q for q in r["questions"]}
    assert qs["Q1A"]["max_marks"] is None and qs["Q1A"]["needs_review"]
    assert qs["Q1B"]["max_marks"] == 4


def test_conflicting_marks_are_flagged():
    q = parse_question_paper([(1, "Q1 Explain paging. [5M] and also [10M]")])["questions"][0]
    assert q["max_marks"] is None and any("Multiple" in r for r in q["review_reasons"])


def test_parent_marks_do_not_silently_split_between_parts():
    r = parse_question_paper([(1, "Q1 [10M]\na) Explain paging.\nb) Explain segmentation.")])
    assert all(q["max_marks"] is None and "parent" in " ".join(q["review_reasons"]) for q in r["questions"])


def test_plain_numbered_questions_and_page_numbers():
    r = parse_question_paper([(1, "1. Define a process. (2 marks)"), (2, "2. Compare TCP and UDP. (5 marks)")])
    assert [(q["label"], q["max_marks"], q["source_page"]) for q in r["questions"]] == [("Q1", 2, 1), ("Q2", 5, 2)]


def test_multiline_question_text_is_joined():
    r = parse_question_paper([(1, "Q1\na) Explain the\nsecond line of text. [3M]")])
    assert r["questions"][0]["text"] == "Explain the second line of text."


def test_no_questions_returns_empty_not_fabricated():
    assert parse_question_paper([(1, "Just some heading text\nwith nothing numbered")])["questions"] == []


@pytest.mark.parametrize("text,marks,expected", [
    ("Define a deadlock.", 2, "definition"), ("Compare TCP and UDP.", 5, "comparison"),
    ("Calculate the average waiting time.", 5, "numerical"), ("Write an algorithm for binary search.", 5, "algorithm"),
    ("Explain virtual memory.", 5, "explanation"), ("Explain virtual memory in detail.", 10, "long_answer"),
    ("What is a process? Discuss its states.", 5, "explanation"), ("Hmm.", 5, "unknown")])
def test_qtype_detection(text, marks, expected):
    assert detect_qtype(text, marks) == expected


# ----------------------------------------------------------------- answer mapping
def test_maps_simple_markers_across_pages():
    pages = [(1, "1a) Distributed system is a collection of computers.\nIt hides distribution.\n1b) Types of transparency are many."),
             (2, "continued text of transparency answer.\n2a) Bully algorithm elects the highest id.")]
    r = map_answers(pages, LABELS)
    assert set(r.segments) == {"Q1A", "Q1B", "Q2A"}
    assert r.segments["Q1B"].pages == [1, 2] and "continued" in r.segments["Q1B"].text


@pytest.mark.parametrize("marker", ["1a)", "Q1 a)", "Q1(a)", "Q.No.1 a)", "Ans 1a:", "1] (a)", "1.a)", "Q1a."])
def test_marker_formats_recognised(marker):
    r = map_answers([(1, f"{marker} The answer text for the first part.")], LABELS)
    assert "Q1A" in r.segments and "answer text" in r.segments["Q1A"].text


def test_heading_then_sub_letters():
    r = map_answers([(1, "Q2\na) Bully text here.\nb) Ring text here.")], LABELS)
    assert set(r.segments) == {"Q2A", "Q2B"}


def test_numbered_list_inside_answer_is_not_a_new_question():
    pages = [(1, "2a) The steps are:\n1. detect failure\n2. send election\n3. choose coordinator\n2b) Ring text.")]
    r = map_answers(pages, LABELS)
    assert "detect failure" in r.segments["Q2A"].text and "choose coordinator" in r.segments["Q2A"].text


def test_no_markers_means_manual_mapping_not_a_guess():
    r = map_answers([(1, "Some text with no question numbers at all.")], LABELS)
    assert r.segments == {} and "Some text" in r.unmapped_text and r.markers_found == 0
    assert any("manually" in n for n in r.notes)


def test_repeated_marker_is_flagged_ambiguous():
    r = map_answers([(1, "1a) first text here.\n1b) other.\n1a) second text here.")], LABELS)
    assert r.segments["Q1A"].occurrences == 2 and any("Q1A" in n for n in r.notes)


def test_unknown_question_marker_is_kept_as_note():
    r = map_answers([(1, "Q9 some stray text\n1a) real answer here.")], LABELS)
    assert "Q1A" in r.segments and any("Q9" in n for n in r.notes)


# ----------------------------------------------------------------- metrics
def test_metrics_perfect_agreement():
    m = compute_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert m["available"] and m["mae"] == 0 and m["rmse"] == 0 and m["exact_agreement"] == 1 and m["pearson"] == 1


def test_metrics_known_values():
    ai, human = [1, 2, 3, 4, 5, 5], [2, 2, 3, 3, 5, 4]
    m = compute_metrics(ai, human, tolerance=1)
    assert m["mae"] == round(3 / 6, 4) and m["rmse"] == round(math.sqrt(3 / 6), 4)
    assert m["exact_agreement"] == round(3 / 6, 4) and m["agreement_within_tolerance"] == 1.0


def test_metrics_refused_for_tiny_samples():
    m = compute_metrics([1, 2], [1, 2])
    assert m["available"] is False and "mae" not in m and m["n"] == 2


def test_metrics_constant_values_have_undefined_correlation():
    m = compute_metrics([2, 2, 2, 2, 2], [1, 2, 3, 4, 5])
    assert m["pearson"] is None and m["spearman"] is None and m["mae"] is not None


def test_metrics_ignore_missing_values():
    m = compute_metrics([1, 2, None, 4, 5, 6], [1, 2, 3, None, 5, 6])
    assert m["n"] == 4 and m["available"] is False


def test_correlation_is_not_agreement():
    ai, human = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]  # perfectly correlated, never equal
    m = compute_metrics(ai, human)
    assert m["pearson"] == 1.0 and m["exact_agreement"] == 0.0 and m["mae"] == 5.0
