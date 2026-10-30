"""End-to-end API tests against the real Flask app + SQLite.

test_full_workflow_*  : OCR is a SCRIPTED MOCK (deterministic) - exercises everything except OCR quality.
test_real_tesseract_* : same pipeline with the REAL Tesseract binary on a synthetic printed 'scan'.
Handwriting OCR through Qwen2-VL is NOT covered by any test in this repository.
"""
import csv
import io
import shutil

import pymupdf as fitz
import pytest

from app import create_app
from app.config import TestConfig
from app.extensions import db
from app.services.ocr import OCRResult, register_engine_for_tests
from tests.conftest import QUESTION_PAPER, upload

ANSWER_PAGE_1 = """1a) A distributed system is a collection of independent computers that appears to its users as a single
coherent system. The goals are resource sharing, transparency, openness and scalability of the system.
1b) Transparency hides the distribution from users. Access transparency hides differences in data
representation and location transparency hides where a resource is physically located."""
ANSWER_PAGE_2 = """2a) The Bully algorithm elects a new coordinator when the current coordinator fails. A process that notices
the failure sends an election message to all processes having a higher id. The process with the highest id wins and
announces itself as the coordinator."""


class ScriptedOCR:
    """MOCK OCR: returns fixed text per call order. Not a statement about real OCR quality."""
    name = "scripted"

    def __init__(self, texts):
        self.texts, self.i = texts, 0

    def recognize(self, image):
        t = self.texts[self.i % len(self.texts)]
        self.i += 1
        return OCRResult(text=t, engine="scripted-mock")


def rubric(core_text, core_kw, detail_text, detail_kw, marks=(2, 3)):
    return {"criteria": [
        {"name": "Core idea", "max_marks": marks[0], "concepts": [{"text": core_text, "keywords": core_kw}]},
        {"name": "Details", "max_marks": marks[1], "concepts": [{"text": detail_text, "keywords": detail_kw}]}]}


RUBRICS = {
    "Q1A": ("A distributed system is a collection of independent computers that appears as a single coherent system",
            ["independent computers", "single", "coherent"],
            "Goals include resource sharing, transparency, openness and scalability",
            ["resource sharing", "transparency", "scalability"]),
    "Q1B": ("Transparency means hiding the distribution of processes and resources from users",
            ["hides", "distribution", "users"],
            "Types of transparency include access and location transparency",
            ["access", "location", "transparency"]),
    "Q2A": ("The Bully algorithm elects a coordinator when the current coordinator fails",
            ["bully", "coordinator", "fails"],
            "A process sends election messages to higher id processes and the highest id wins",
            ["election", "higher", "highest"]),
}


def j(resp):
    return resp.get_json()


def setup_exam(client, paper_pdf):
    exam = j(client.post("/api/exams", json={"name": "IA1 Distributed Systems", "subject": "DS",
                                             "choice_groups": [["Q1", "Q2"]]}))
    r = upload(client, f"/api/exams/{exam['id']}/question-paper", paper_pdf, "paper.pdf")
    assert r.status_code == 201, r.get_json()
    return exam["id"], j(r)["questions"]


def configure_grading(client, questions):
    by_label = {q["label"]: q for q in questions}
    for label, spec in RUBRICS.items():
        q = by_label[label]
        r = client.put(f"/api/questions/{q['id']}/reference", json={"text": spec[0] + ". " + spec[2] + "."})
        assert r.status_code == 200
        r = client.put(f"/api/questions/{q['id']}/rubric", json=rubric(*spec))
        assert r.status_code == 200, r.get_json()
    return by_label


@pytest.fixture()
def scripted_ocr():
    eng = ScriptedOCR([ANSWER_PAGE_1, ANSWER_PAGE_2])
    register_engine_for_tests("tesseract", eng)
    yield eng
    register_engine_for_tests("tesseract", __import__("app.services.ocr", fromlist=["TesseractEngine"]).TesseractEngine())


def test_full_workflow_with_mocked_ocr(client, pdfs, scripted_ocr):
    # 1-3: exam + question paper (text PDF -> direct extraction, no OCR)
    exam_id, questions = setup_exam(client, pdfs.text(QUESTION_PAPER.splitlines() and [QUESTION_PAPER]))
    assert [q["label"] for q in questions] == ["Q1A", "Q1B", "Q2A", "Q2B", "Q3A"]
    assert all(q["max_marks"] == 5 and q["source_page"] == 1 for q in questions)

    # 4: correct an extracted question
    q1a = questions[0]
    r = client.patch(f"/api/questions/{q1a['id']}", json={"text": q1a["text"] + " (edited)"})
    assert j(r)["text"].endswith("(edited)")
    # add a missing one manually
    r = client.post(f"/api/exams/{exam_id}/questions", json={"label": "q3b", "text": "Define clock drift.", "max_marks": 2})
    assert r.status_code == 201 and j(r)["label"] == "Q3B" and j(r)["qtype"] == "definition"

    # 5: references + rubrics (Q2B, Q3A, Q3B deliberately left without a rubric)
    by_label = configure_grading(client, questions)

    # rubric validation: criteria must add up to the question's marks
    bad = rubric("x", ["x"], "y", ["y"], marks=(1, 1))
    assert client.put(f"/api/questions/{by_label['Q1A']['id']}/rubric", json=bad).status_code == 422

    # 6-7: answer sheet upload + OCR (2 scanned pages)
    sheet = pdfs.image([pdfs.text_image(["page one"]), pdfs.text_image(["page two"])])
    r = upload(client, f"/api/exams/{exam_id}/submissions", sheet, "sheet.pdf", student_identifier="2SD23CS082")
    assert r.status_code == 201, r.get_json()
    sub = j(r)["submission"]
    sid = sub["id"]
    assert sub["status"] == "mapped" and [p["text_original"][:2] for p in sub["pages"]] == ["1a", "2a"]
    assert scripted_ocr.i == 2
    assert all(p["confidence"] is None for p in sub["pages"])  # mock gave none; none invented

    # 8: reviewer corrects OCR on page 2 and marks it reviewed, then re-maps
    page2 = sub["pages"][1]["text_original"]
    r = client.patch(f"/api/submissions/{sid}/pages/2", json={"text_corrected": page2.replace("Bully", "Bully"), "reviewed": True})
    assert r.status_code == 200
    sub = j(client.post(f"/api/submissions/{sid}/map"))
    answers = {a["question_label"]: a for a in sub["answers"]}
    assert answers["Q1A"]["status"] == "answered" and "distributed system" in answers["Q1A"]["text_original"]
    assert answers["Q2A"]["pages"] == [2]
    # unanswered questions are distinguishable (no page failed OCR)
    assert answers["Q3A"]["status"] == "unanswered" and answers["Q2B"]["status"] == "unanswered"

    # 9-12: evaluate
    r = client.post(f"/api/submissions/{sid}/evaluate", json={})
    assert r.status_code == 202 and j(r)["job"]["status"] == "done"
    sub = j(client.get(f"/api/submissions/{sid}"))
    ev = {a["question_label"]: a["evaluation"] for a in sub["answers"]}
    for label in ("Q1A", "Q1B", "Q2A"):
        e = ev[label]
        assert e["ai_total"] is not None and 0 <= e["ai_total"] <= 5, (label, e)
        assert e["ai_result"]["criteria"] and e["ai_result"]["awarded_because"] is not None
        assert e["ai_error"] is None
    assert ev["Q1A"]["ai_total"] >= 4  # strong answer, rubric keywords all present
    # no rubric -> manual_required with NO marks (never a fabricated zero or guess)
    assert ev["Q3A"]["status"] == "manual_required" and ev["Q3A"]["ai_total"] is None
    assert ev["Q2B"]["status"] == "manual_required" and "rubric" in ev["Q2B"]["ai_error"].lower() or "reference" in ev["Q2B"]["ai_error"].lower()

    # 13: human review - override Q1A, approve Q1B/Q2A as AI scored
    r = client.patch(f"/api/evaluations/{ev['Q1A']['id']}/review", json={"total": 3.5, "comment": "Missed openness", "approve": True})
    assert r.status_code == 200 and j(r)["final_total"] == 3.5 and j(r)["ai_total"] == ev["Q1A"]["ai_total"]
    assert j(r)["human_total"] == 3.5  # human value stored separately from the AI value
    assert client.patch(f"/api/evaluations/{ev['Q1A']['id']}/review", json={"total": 9}).status_code == 422
    assert client.patch(f"/api/evaluations/{ev['Q1A']['id']}/review", json={"criterion_scores": {"99999": 1}}).status_code == 422
    for label in ("Q1B", "Q2A"):
        assert client.patch(f"/api/evaluations/{ev[label]['id']}/review", json={"approve": True}).status_code == 200
    # manual grading of a question the AI could not grade
    r = client.patch(f"/api/evaluations/{ev['Q3A']['id']}/review", json={"approve": True})
    assert r.status_code == 422  # cannot approve with no score at all
    r = client.patch(f"/api/evaluations/{ev['Q3A']['id']}/review", json={"total": 0, "comment": "Not attempted", "approve": True})
    assert r.status_code == 200 and j(r)["final_total"] == 0

    # 14: results apply the choice rule (best of Q1/Q2 counts) and the exam maximum
    res = j(client.get(f"/api/submissions/{sid}/results"))
    rows = {r["label"]: r for r in res["questions"]}
    q1 = rows["Q1A"]["marks"] + rows["Q1B"]["marks"]
    q2 = rows["Q2A"]["marks"]
    assert res["max_total"] == 10 + 5 + 2  # best of (Q1=10, Q2=10) + Q3A(5) + Q3B(2)
    assert res["total"] == pytest.approx(max(q1, q2) + rows["Q3A"]["marks"])
    assert res["total"] <= res["max_total"]

    # 15: exports reflect the saved data exactly
    csv_text = client.get(f"/api/submissions/{sid}/report.csv").get_data(as_text=True)
    rows_csv = {r["question"]: r for r in csv.DictReader(io.StringIO(csv_text))}
    assert rows_csv["Q1A"]["ai_marks"] == f"{ev['Q1A']['ai_total']:g}"
    assert rows_csv["Q1A"]["human_marks"] == "3.5" and rows_csv["Q1A"]["final_marks"] == "3.5"
    assert rows_csv["Q1A"]["marks_type"] == "human-approved"
    assert rows_csv["Q2B"]["final_marks"] == "" and rows_csv["Q2B"]["approved"] == "no"
    pdf_resp = client.get(f"/api/submissions/{sid}/report.pdf")
    assert pdf_resp.status_code == 200 and pdf_resp.data.startswith(b"%PDF")
    text = "".join(p.get_text() for p in fitz.open(stream=pdf_resp.data, filetype="pdf"))
    assert "2SD23CS082" in text and "Missed openness" in text and f"{res['total']:g} / {res['max_total']:g}" in text
    assert "human-approved" in text or "Final: 3.5" in text
    exam_csv = client.get(f"/api/exams/{exam_id}/report.csv").get_data(as_text=True)
    assert exam_csv.count("2SD23CS082") == len(questions) + 1

    # metrics: only 3 human-reviewed pairs < min samples -> refused rather than misleading
    m = j(client.get(f"/api/exams/{exam_id}/metrics"))
    assert m["available"] is False

    # dashboard + history reflect reality
    d = j(client.get("/api/dashboard"))
    assert d["submission_count"] == 1 and d["recent"][0]["student_identifier"] == "2SD23CS082"
    hist = j(client.get(f"/api/submissions?exam_id={exam_id}"))
    assert hist[0]["total"] == res["total"]


def test_upload_validation(client, pdfs):
    exam = j(client.post("/api/exams", json={"name": "E"}))
    url = f"/api/exams/{exam['id']}/question-paper"
    assert upload(client, url, b"hello", "notes.txt").status_code == 400
    assert upload(client, url, b"hello", "fake.pdf").status_code == 400            # wrong content
    assert upload(client, url, pdfs.corrupt, "bad.pdf").status_code == 400
    assert upload(client, url, b"", "empty.pdf").status_code == 400
    assert client.post(url, data={}, content_type="multipart/form-data").status_code == 400
    assert upload(client, url, pdfs.text(["ok page with enough characters on it to be text"]), "../../etc/x.pdf").status_code in (201, 202)


def test_oversized_upload_rejected(app, client, pdfs):
    app.config["MAX_UPLOAD_MB"] = 1
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    exam = j(client.post("/api/exams", json={"name": "E"}))
    big = pdfs.text(["x" * 50]) + b"\n%" + b"0" * (2 * 1024 * 1024)
    assert upload(client, f"/api/exams/{exam['id']}/question-paper", big, "big.pdf").status_code == 413


def test_stored_filenames_are_server_generated(app, client, pdfs):
    exam = j(client.post("/api/exams", json={"name": "E"}))
    r = upload(client, f"/api/exams/{exam['id']}/question-paper", pdfs.text(["a page with enough text on it to count"]), "../../evil name.pdf")
    doc = j(r)["document"]
    assert "/" not in doc["filename"] and ".." not in doc["filename"].replace("...", "")
    import os
    names = os.listdir(app.config["UPLOAD_DIR"])
    assert len(names) == 1 and len(names[0]) == 36


def test_reference_ai_draft_must_be_approved_before_grading(client, pdfs, scripted_ocr):
    exam_id, questions = setup_exam(client, pdfs.text([QUESTION_PAPER]))
    q = questions[0]
    r = client.post(f"/api/questions/{q['id']}/reference/draft")
    assert r.status_code == 503  # no external LLM configured: explicit, not silently faked
    # simulate an unapproved AI draft
    from app.models import Question
    row = db.session.get(Question, q["id"])
    row.reference_answer, row.reference_source, row.reference_approved = "Draft text", "ai_draft", False
    db.session.commit()
    spec = RUBRICS["Q1A"]
    assert client.put(f"/api/questions/{q['id']}/rubric", json=rubric(*spec)).status_code == 200
    sheet = pdfs.image([pdfs.text_image(["p1"])])
    sid = j(upload(client, f"/api/exams/{exam_id}/submissions", sheet, "s.pdf"))["submission"]["id"]
    client.post(f"/api/submissions/{sid}/evaluate", json={})
    ev = next(a["evaluation"] for a in j(client.get(f"/api/submissions/{sid}"))["answers"] if a["question_label"] == "Q1A")
    assert ev["status"] == "manual_required" and "unapproved" in ev["ai_error"]
    client.post(f"/api/questions/{q['id']}/reference/approve")
    client.post(f"/api/submissions/{sid}/evaluate", json={})
    ev = next(a["evaluation"] for a in j(client.get(f"/api/submissions/{sid}"))["answers"] if a["question_label"] == "Q1A")
    assert ev["ai_total"] is not None


def test_ambiguous_mapping_requires_confirmation(client, pdfs):
    eng = ScriptedOCR(["Some answer text without any question numbers about distributed systems and resource sharing."])
    register_engine_for_tests("tesseract", eng)
    exam = j(client.post("/api/exams", json={"name": "One question"}))
    client.post(f"/api/exams/{exam['id']}/questions", json={"label": "Q1", "text": "Explain distributed systems.", "max_marks": 5})
    sid = j(upload(client, f"/api/exams/{exam['id']}/submissions", pdfs.image([pdfs.text_image(["x"])]), "s.pdf"))["submission"]["id"]
    ans = j(client.get(f"/api/submissions/{sid}"))["answers"][0]
    assert ans["status"] == "ambiguous" and ans["mapping_confirmed"] is False
    r = client.patch(f"/api/answers/{ans['id']}", json={"mapping_confirmed": True})
    assert j(r)["status"] == "answered"


def test_ocr_failure_gives_unreadable_not_unanswered(client, pdfs):
    class Broken:
        name = "broken"

        def recognize(self, image):
            raise RuntimeError("model crashed")
    register_engine_for_tests("tesseract", Broken())
    exam = j(client.post("/api/exams", json={"name": "E"}))
    client.post(f"/api/exams/{exam['id']}/questions", json={"label": "Q1", "text": "Explain x.", "max_marks": 5})
    r = upload(client, f"/api/exams/{exam['id']}/submissions", pdfs.image([pdfs.text_image(["x"])]), "s.pdf")
    sub = j(r)["submission"]
    assert sub["status"] == "ocr_failed" and sub["pages"][0]["error"] and sub["pages"][0]["text_original"] == ""
    assert j(client.post(f"/api/submissions/{sub['id']}/map"))["answers"][0]["status"] == "unreadable"


def test_settings_validation(client):
    assert client.put("/api/settings", json={"ocr_engine": "magic"}).status_code == 422
    assert client.put("/api/settings", json={"credit_correct": 3}).status_code == 422
    assert client.put("/api/settings", json={"nonsense": 1}).status_code == 422
    ok = client.put("/api/settings", json={"mark_tolerance": 1, "ocr_dpi": 150})
    assert ok.status_code == 200 and j(ok)["settings"]["ocr_dpi"] == 150


def test_page_image_endpoint_renders_real_png(client, pdfs):
    exam = j(client.post("/api/exams", json={"name": "E"}))
    doc = j(upload(client, f"/api/exams/{exam['id']}/question-paper", pdfs.text(["Q1 a) Explain x. [5M]"]), "p.pdf"))["document"]
    r = client.get(f"/api/documents/{doc['id']}/pages/1/image?dpi=72")
    assert r.status_code == 200 and r.data[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/documents/{doc['id']}/pages/9/image").status_code == 404


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_real_tesseract_end_to_end_on_synthetic_printed_scan(client, pdfs):
    """REAL OCR (Tesseract) on a synthetic PRINTED scan - not handwriting."""
    from app.services.ocr import TesseractEngine
    register_engine_for_tests("tesseract", TesseractEngine())
    exam_id, questions = setup_exam(client, pdfs.text([QUESTION_PAPER]))
    configure_grading(client, questions)
    page = pdfs.text_image([
        "1a) A distributed system is a collection of independent computers",
        "that appears as a single coherent system. Goals include resource",
        "sharing, transparency, openness and scalability.",
        "1b) Transparency hides the distribution from users. Access and",
        "location transparency are two types of transparency."], font_size=40)
    r = upload(client, f"/api/exams/{exam_id}/submissions", pdfs.image([page]), "scan.pdf", student_identifier="S1")
    sub = j(r)["submission"]
    assert sub["pages"][0]["method"] == "tesseract" and sub["pages"][0]["confidence"] > 50
    answers = {a["question_label"]: a for a in sub["answers"]}
    assert answers["Q1A"]["status"] == "answered" and "distributed" in answers["Q1A"]["text_original"].lower()
    assert any("not been reviewed" in n for n in answers["Q1A"]["notes"])
    client.post(f"/api/submissions/{sub['id']}/evaluate", json={})
    ev = next(a["evaluation"] for a in j(client.get(f"/api/submissions/{sub['id']}"))["answers"] if a["question_label"] == "Q1A")
    assert ev["ai_total"] is not None and ev["ai_total"] >= 3
    # OCR output not yet human-reviewed -> flagged for review, never silently 'ai_graded'
    assert ev["status"] == "needs_review" and any("not been reviewed" in f for f in ev["ai_flags"])


def test_saved_evaluation_survives_restart(tmp_path, pdfs):
    """A second app instance on the same database file sees the data saved by the first."""
    db_uri = f"sqlite:///{tmp_path / 'persist.db'}"

    class Cfg(TestConfig):
        SQLALCHEMY_DATABASE_URI = db_uri
        UPLOAD_DIR = str(tmp_path / "up")

    register_engine_for_tests("tesseract", ScriptedOCR([ANSWER_PAGE_1]))
    app1 = create_app(Cfg)
    with app1.app_context():
        db.create_all()
        c = app1.test_client()
        exam_id, qs = setup_exam(c, pdfs.text([QUESTION_PAPER]))
        configure_grading(c, qs)
        sid = j(upload(c, f"/api/exams/{exam_id}/submissions", pdfs.image([pdfs.text_image(["x"])]), "s.pdf",
                       student_identifier="PERSIST"))["submission"]["id"]
        c.post(f"/api/submissions/{sid}/evaluate", json={})
        ev = next(a["evaluation"] for a in j(c.get(f"/api/submissions/{sid}"))["answers"] if a["question_label"] == "Q1A")
        c.patch(f"/api/evaluations/{ev['id']}/review", json={"total": 4, "comment": "saved", "approve": True})
        before = j(c.get(f"/api/submissions/{sid}/results"))
        db.session.remove()
        db.engine.dispose()

    app2 = create_app(Cfg)  # "restart"
    with app2.app_context():
        c2 = app2.test_client()
        after = j(c2.get(f"/api/submissions/{sid}/results"))
        assert after == before
        ev2 = next(a["evaluation"] for a in j(c2.get(f"/api/submissions/{sid}"))["answers"] if a["question_label"] == "Q1A")
        assert ev2["final_total"] == 4 and ev2["reviewer_comment"] == "saved" and ev2["approved"]
        assert c2.get(f"/api/submissions/{sid}/report.pdf").data.startswith(b"%PDF")
        assert j(c2.get("/api/exams"))[0]["name"] == "IA1 Distributed Systems"
        db.session.remove()
        db.engine.dispose()
