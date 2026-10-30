"""Grading engine tests. No Flask, no DB. Embedder = lexical fallback unless noted.

Tests marked 'scripted judge' use a fake judge to exercise validation/scoring logic deterministically;
they do not prove anything about the heuristic judge's quality.
"""
import importlib.util

import pytest

from app.grading.embeddings import LexicalEmbedder, get_embedder
from app.grading.engine import GradingEngine, compute_marks, precheck
from app.grading.judges import HeuristicJudge, JudgeError
from app.grading.schema import AnswerSpec, ConceptSpec, CriterionSpec, QuestionSpec
from app.grading.validation import ValidationFailure, validate_judgement


def make_question(qtype="explanation", approved=True, reference="An index speeds up retrieval of rows."):
    return QuestionSpec(
        1, "Q1", "Explain database indexing.", qtype, 5.0, reference, approved,
        [CriterionSpec(1, "Core concept", 1, concepts=[ConceptSpec(
            "c1", "An index is a data structure that speeds up retrieval of rows",
            ["index", "data structure", "retrieval"], ["lookup", "search"])]),
         CriterionSpec(2, "Mechanism", 2, concepts=[
             ConceptSpec("c2", "A B-tree lets the database find rows without scanning the whole table",
                         ["b-tree", "scan"], ["balanced tree"]),
             ConceptSpec("c3", "Index entries point to table rows through a key", ["pointer", "key"], [])]),
         CriterionSpec(3, "Trade-offs", 1, concepts=[ConceptSpec(
             "c4", "Indexes use extra storage and slow down inserts and updates", ["storage", "slow", "writes"], [])]),
         CriterionSpec(4, "Example", 1, concepts=[ConceptSpec(
             "c5", "An example such as indexing a primary key column", ["example", "primary key", "column"], [])])])


@pytest.fixture(scope="module")
def engine():
    emb = LexicalEmbedder()
    return GradingEngine(HeuristicJudge(emb), embedder_name=emb.name)


FULL = ("An index is a data structure that makes retrieval of rows much faster. The database walks a B-tree so it can "
        "find rows without scanning the whole table. Each entry holds a key and a pointer to the row. The cost is extra "
        "storage and slower inserts and updates. For example we index the primary key column.")


def test_fully_correct_answer_gets_full_marks(engine):
    o = engine.evaluate(make_question(), AnswerSpec(FULL))
    assert o.result.total == 5.0
    assert all(c.status == "correct" for cr in o.result.criteria for c in cr.concepts)
    assert o.status == "ai_graded" and not o.flags


def test_valid_synonym_wording_is_accepted(engine):
    answer = ("An index is a data structure for quick search of rows. A balanced tree lets the database find rows "
              "without scanning the whole table. Each entry holds a key and a pointer to the row. Extra storage is "
              "needed and writes slow down. For example the primary key column is indexed.")
    o = engine.evaluate(make_question(), AnswerSpec(answer))
    assert o.result.total >= 4.5


def test_partial_answer_gets_partial_credit_and_names_missing_concepts(engine):
    o = engine.evaluate(make_question(), AnswerSpec(
        "An index is a data structure that makes retrieval of rows much faster for the database."))
    assert 0 < o.result.total < 5
    text = " ".join(o.result.improvements)
    assert "B-tree" in text and "extra storage" in text.lower()
    assert any("missing" in w for w in o.result.withheld_because)


def test_keywords_without_explanation_only_get_mentioned_credit(engine):
    o = engine.evaluate(make_question(), AnswerSpec("index b-tree storage example"))
    statuses = {c.concept_id: c.status for cr in o.result.criteria for c in cr.concepts}
    assert "mentioned" in statuses.values() and "correct" not in statuses.values()
    assert o.result.total < 2.5
    assert any("not explained" in w for w in o.result.withheld_because)


def test_irrelevant_answer_scores_zero(engine):
    o = engine.evaluate(make_question(), AnswerSpec("The French revolution began in 1789 and reshaped Europe politically."))
    assert o.result.total == 0.0


def test_length_does_not_buy_marks(engine):
    padding = " ".join(["The weather today is pleasant and the garden is green."] * 40)
    o = engine.evaluate(make_question(), AnswerSpec(padding))
    assert o.result.total == 0.0


def test_grammar_and_spelling_not_penalised(engine):
    messy = FULL.replace("retrieval", "retreival ").lower().replace(".", "")
    base = engine.evaluate(make_question(), AnswerSpec(FULL)).result.total
    assert engine.evaluate(make_question(), AnswerSpec(messy)).result.total >= base - 1.0


def test_contradictory_statement_is_not_credited_and_is_flagged(engine):
    q = make_question()
    o = engine.evaluate(q, AnswerSpec("An index is not a data structure and does not speed up retrieval of rows at all."))
    c1 = o.result.criteria[0].concepts[0]
    assert c1.status == "incorrect"
    assert o.result.criteria[0].awarded == 0
    assert o.status == "needs_review" and any("possibly_incorrect" in f for f in o.flags)


def test_unanswered_question_is_zero_but_flagged_for_confirmation(engine):
    o = engine.evaluate(make_question(), AnswerSpec("", status="unanswered"))
    assert o.result.total == 0 and o.status == "needs_review"
    assert any("unanswered" in f for f in o.flags)


def test_unreadable_answer_is_manual_not_zero(engine):
    o = engine.evaluate(make_question(), AnswerSpec("", status="unreadable"))
    assert o.status == "manual_required" and o.result is None and "OCR" in o.error


def test_unconfirmed_ambiguous_mapping_blocks_grading(engine):
    o = engine.evaluate(make_question(), AnswerSpec(FULL, status="ambiguous", mapping_confirmed=False))
    assert o.status == "manual_required" and o.result is None


def test_unapproved_ai_reference_blocks_grading(engine):
    o = engine.evaluate(make_question(approved=False), AnswerSpec(FULL))
    assert o.status == "manual_required" and "unapproved" in o.error


def test_missing_reference_blocks_grading(engine):
    o = engine.evaluate(make_question(reference=""), AnswerSpec(FULL))
    assert o.status == "manual_required" and "reference" in o.error.lower()


def test_numerical_questions_are_routed_to_manual_grading(engine):
    o = engine.evaluate(make_question(qtype="numerical"), AnswerSpec("x = 42"))
    assert o.status == "manual_required" and o.result is None


def test_rubric_not_adding_up_blocks_grading(engine):
    q = make_question()
    q.criteria[0].max_marks = 3
    assert "add up" in precheck(q, AnswerSpec(FULL), engine.judge)


def test_multiple_subquestions_are_graded_independently(engine):
    q1, q2 = make_question(), make_question()
    q2.id, q2.label = 2, "Q1B"
    a = engine.evaluate(q1, AnswerSpec(FULL))
    b = engine.evaluate(q2, AnswerSpec("Unrelated text about cooking pasta in salted boiling water."))
    assert a.result.question_id == 1 and b.result.question_id == 2
    assert a.result.total > b.result.total == 0


def test_marks_never_exceed_limits(engine):
    o = engine.evaluate(make_question(), AnswerSpec(FULL))
    assert o.result.total <= o.result.max_total
    assert all(0 <= c.awarded <= c.max_marks for c in o.result.criteria)


def test_compute_marks_clamps_even_with_bad_credits():
    q = make_question()
    assessed = [{"criterion_id": c.id, "concept_id": k.id, "status": "correct", "evidence": "x", "comment": "y"}
                for c in q.criteria for k in c.concepts]
    criteria, total = compute_marks(q, assessed, {"correct": 5.0, "partial": 0, "mentioned": 0, "incorrect": 0, "missing": 0})
    assert all(c.awarded <= c.max_marks for c in criteria) and total <= q.max_marks


# ------------------------------------------------------------------ scripted-judge validation tests
class ScriptedJudge:
    name = "scripted"
    supported_qtypes = {"explanation"}

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def assess(self, question, answer):
        self.calls += 1
        out = self.outputs[min(self.calls - 1, len(self.outputs) - 1)]
        if isinstance(out, Exception):
            raise out
        return out


def good_output(q, answer_text, status="correct"):
    ev = answer_text.split(".")[0]
    return {"concepts": [{"criterion_id": c.id, "concept_id": k.id, "status": status, "evidence": ev,
                          "comment": "assessed"} for c in q.criteria for k in c.concepts]}


def test_invalid_ai_output_is_retried_then_sent_to_manual_review():
    q = make_question()
    judge = ScriptedJudge([{"concepts": [{"criterion_id": 99, "concept_id": "zz", "status": "correct"}]}])
    o = GradingEngine(judge, max_attempts=3).evaluate(q, AnswerSpec(FULL))
    assert judge.calls == 3 and o.status == "manual_required" and o.result is None
    assert o.attempts == 3 and "validation" in o.error


def test_retry_recovers_when_second_attempt_is_valid():
    q = make_question()
    judge = ScriptedJudge([{"concepts": []}, good_output(q, FULL)])
    o = GradingEngine(judge, max_attempts=2).evaluate(q, AnswerSpec(FULL))
    assert judge.calls == 2 and o.result.total == 5.0


def test_judge_outage_is_manual_required_not_a_fabricated_score():
    o = GradingEngine(ScriptedJudge([JudgeError("connection refused")])).evaluate(make_question(), AnswerSpec(FULL))
    assert o.status == "manual_required" and o.result is None and "connection refused" in o.error


def test_valid_alternative_explanation_is_credited_when_judge_finds_it():
    q = make_question()
    alt = "Think of an index like a book's table of contents: it lets you jump straight to the rows you want."
    out = good_output(q, alt, "correct")
    o = GradingEngine(ScriptedJudge([out])).evaluate(q, AnswerSpec(alt))
    assert o.result.total == 5.0


@pytest.mark.parametrize("mutate,expected", [
    (lambda r: r["concepts"][0].update(status="excellent"), "invalid status"),
    (lambda r: r["concepts"][0].update(evidence="a quote that is not in the answer"), "not a quote"),
    (lambda r: r["concepts"][0].update(evidence=None), "requires evidence"),
    (lambda r: r["concepts"][0].update(comment=""), "explanation"),
    (lambda r: r["concepts"].pop(), "was not assessed"),
    (lambda r: r["concepts"].append(dict(r["concepts"][0])), "more than once"),
    (lambda r: r["concepts"][0].update(criterion_id=999), "unknown criterion_id"),
])
def test_validation_rejects_bad_output(mutate, expected):
    q = make_question()
    raw = good_output(q, FULL)
    mutate(raw)
    with pytest.raises(ValidationFailure) as exc:
        validate_judgement(raw, q, AnswerSpec(FULL))
    assert expected in str(exc.value)


def test_validation_rejects_non_object():
    with pytest.raises(ValidationFailure):
        validate_judgement("just text", make_question(), AnswerSpec(FULL))


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is None,
                    reason="sentence-transformers not installed here; semantic paraphrase quality is UNTESTED")
def test_semantic_paraphrase_without_shared_vocabulary():
    emb = get_embedder("sentence-transformers")
    o = GradingEngine(HeuristicJudge(emb), embedder_name=emb.name).evaluate(
        make_question(), AnswerSpec("A lookup structure that lets the DBMS locate records quickly instead of reading everything."))
    assert o.result.criteria[0].concepts[0].status in ("correct", "partial")
