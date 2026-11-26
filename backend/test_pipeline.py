import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pymupdf
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

import database

_test_storage = tempfile.TemporaryDirectory(prefix="bert-grading-tests-")
_test_root = Path(_test_storage.name)
database.DB_NAME = str(_test_root / "grading_system.db")
database.DATA_DIR = _test_root / "data"
database.TRAINING_LOG = database.DATA_DIR / "training_dataset.jsonl"

import main
from bert_scorer import evaluate_with_bert
from chunker import chunk_student_booklet, convert_pdf_to_images
from handwriting_ocr import segment_handwriting_lines


def _mock_evaluation(**kwargs):
    return {
        "matched_points": ["Research is systematic"],
        "missing_points": [],
        "ai_score": 4.5,
        "similarity_score": 0.91,
        "keyword_coverage": 0.8,
        "ai_feedback": "Strong answer with clear explanation.",
    }


class OCRPipelineTests(unittest.TestCase):
    def test_pdf_conversion_preserves_page_order(self):
        document = pymupdf.open()
        document.new_page().insert_text((50, 50), "first page")
        document.new_page().insert_text((50, 50), "second page")
        pdf_bytes = document.tobytes()
        document.close()

        with tempfile.TemporaryDirectory() as output_dir:
            pages = convert_pdf_to_images(pdf_bytes, output_dir)
            self.assertEqual(len(pages), 2)
            self.assertTrue(Path(pages[0]).name.endswith("1.png"))
            self.assertTrue(Path(pages[1]).name.endswith("2.png"))
            self.assertTrue(all(Path(page).is_file() for page in pages))

    def test_opencv_segments_separate_handwriting_lines(self):
        image = np.full((320, 1200), 255, dtype=np.uint8)
        cv2.putText(image, "Q1A Research is systematic", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        cv2.putText(image, "It uses a clear method", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

        with tempfile.TemporaryDirectory() as output_dir:
            image_path = Path(output_dir) / "page.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            line_images = segment_handwriting_lines(str(image_path))
            self.assertEqual(len(line_images), 2)

    def test_chunker_detects_question_ids_beyond_original_range(self):
        rubrics = [
            {"question_id": "Q1A", "question_text": "Answer one"},
            {"question_id": "Q2B", "question_text": "Answer two"},
            {"question_id": "Q4C", "question_text": "Answer three"},
            {"question_id": "Q10D", "question_text": "Answer four"},
        ]
        transcript = "\n".join(
            [
                "Q1A) First answer text.",
                "Q2B) Second answer text.",
                "Q4C) Third answer text.",
                "Question 10 (d): Fourth answer text.",
                "Q99B) Unmapped answer text.",
            ]
        )

        chunks = chunk_student_booklet(transcript, rubrics)

        self.assertEqual(set(chunks), {"Q1A", "Q2B", "Q4C", "Q10D", "Q99B"})
        self.assertIn("Unmapped answer", chunks["Q99B"])


class BertScoringTests(unittest.TestCase):
    def test_bert_rubric_score_is_bounded_and_returns_metrics(self):
        def fake_encode(texts):
            vectors = []
            for text in texts:
                words = set(text.lower().split())
                vector = np.array(
                    [
                        float("research" in words),
                        float("systematic" in words),
                        float("method" in words),
                        float("unrelated" in words),
                    ],
                    dtype=np.float32,
                )
                norm = np.linalg.norm(vector)
                vectors.append(vector / norm if norm else vector)
            return np.stack(vectors)

        with patch("bert_scorer._encode_texts", side_effect=fake_encode):
            result = evaluate_with_bert(
                question_text="Explain research",
                reference_answer="Research is a systematic method",
                max_marks=5,
                key_points=[
                    {"point": "Research is systematic", "weight": 2.5},
                    {"point": "A research method", "weight": 2.5},
                ],
                student_answer="Research is systematic and uses a method.",
            )

        self.assertGreaterEqual(result["ai_score"], 0)
        self.assertLessEqual(result["ai_score"], 5)
        self.assertGreater(result["similarity_score"], 0)
        self.assertGreater(result["keyword_coverage"], 0)
        self.assertTrue(result["ai_feedback"])


class APIGradingTests(unittest.TestCase):
    def setUp(self):
        database.DB_NAME = str(_test_root / f"grading_{self._testMethodName}.db")
        main.init_db()
        self.client = TestClient(main.app)
        response = self.client.post(
            "/api/exams",
            json={
                "exam_id": "BERT_TEST",
                "exam_title": "BERT test",
                "questions": [
                    {
                        "question_id": "Q1A",
                        "question_text": "Explain research methods",
                        "max_marks": 5,
                        "key_points": [{"point": "Research is systematic", "weight": 5}],
                        "reference_answer": "Research is a systematic method of inquiry.",
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)

    def _pdf_bytes(self):
        document = pymupdf.open()
        document.new_page().insert_text((50, 50), "sample answer")
        content = document.tobytes()
        document.close()
        return content

    def test_batch_pdf_job_and_individual_question_grading(self):
        transcript = "Q1A) Research is systematic and uses a method."
        with (
            patch.object(main.engine, "transcribe_images", return_value=transcript),
            patch.object(main.engine, "evaluate_answer", side_effect=_mock_evaluation) as evaluate,
            patch.object(main.engine, "unload_model"),
        ):
            response = self.client.post(
                "/api/exams/BERT_TEST/grade",
                data={"student_id": "STUDENT1"},
                files=[("files", ("answers.pdf", self._pdf_bytes(), "application/pdf"))],
            )
            self.assertEqual(response.status_code, 200, response.text)
            job = self.client.get(f"/api/jobs/{response.json()['job_id']}").json()
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["chunks"], ["Q1A"])
            self.assertEqual(job["results"][0]["ai_score"], 4.5)
            self.assertEqual(job["results"][0]["similarity_score"], 0.91)

            graded = self.client.post(
                "/api/submissions/evaluate-student",
                json={
                    "exam_id": "BERT_TEST",
                    "student_id": "STUDENT1",
                    "question_id": "Q1A",
                },
            )
            self.assertEqual(graded.status_code, 200, graded.text)
            self.assertEqual(graded.json()["submissions"][0]["ai_score"], 4.5)
            self.assertEqual(evaluate.call_count, 2)

    def test_rubric_api_round_trips_reference_answer(self):
        response = self.client.get("/api/exams/BERT_TEST/rubrics")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json()[0]["reference_answer"],
            "Research is a systematic method of inquiry.",
        )

    def test_batch_grades_all_detected_answers_and_reports_unmapped_sections(self):
        questions = [
            {
                "question_id": qid,
                "question_text": f"Explain {qid}",
                "max_marks": 5,
                "key_points": ["clear explanation"],
            }
            for qid in ("Q1A", "Q2B", "Q4C", "Q10D")
        ]
        response = self.client.post(
            "/api/exams",
            json={"exam_id": "MULTI_TEST", "exam_title": "Multi answer test", "questions": questions},
        )
        self.assertEqual(response.status_code, 200, response.text)

        transcript = "\n".join(
            [
                "Q1A) First answer text.",
                "Q2B) Second answer text.",
                "Q4C) Third answer text.",
                "Q10D) Fourth answer text.",
                "Q99B) Unmapped answer text.",
            ]
        )
        with (
            patch.object(main.engine, "transcribe_images", return_value=transcript),
            patch.object(main.engine, "evaluate_answer", side_effect=_mock_evaluation) as evaluate,
            patch.object(main.engine, "unload_model"),
        ):
            response = self.client.post(
                "/api/exams/MULTI_TEST/grade",
                data={"student_id": "STUDENT2"},
                files=[("files", ("answers.pdf", self._pdf_bytes(), "application/pdf"))],
            )
            self.assertEqual(response.status_code, 200, response.text)
            job = self.client.get(f"/api/jobs/{response.json()['job_id']}").json()

            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(
                {item["question_id"] for item in job["results"]},
                {"Q1A", "Q2B", "Q4C", "Q10D"},
            )
            self.assertEqual(job["diagnostics"]["answers_detected_by_segmentation"], 5)
            self.assertEqual(job["diagnostics"]["answers_passed_to_grader"], 4)
            self.assertEqual(job["diagnostics"]["answers_evaluated_successfully"], 4)
            self.assertTrue(job["needs_review"])
            self.assertEqual(job["diagnostics"]["skipped"][0]["question_id"], "Q99B")
            self.assertEqual(job["total_score"], 18.0)
            self.assertEqual(job["max_possible_score"], 20.0)

            with patch.object(
                main.engine,
                "evaluate_answer",
                side_effect=lambda **kwargs: {**_mock_evaluation(), "ai_score": 99.0},
            ):
                manual = self.client.post(
                    "/api/exams/MULTI_TEST/grade-manual",
                    json={
                        "exam_id": "MULTI_TEST",
                        "student_id": "STUDENT2",
                        "answers": [
                            {"question_id": qid, "answer": f"Corrected answer for {qid}"}
                            for qid in ("Q1A", "Q2B", "Q4C", "Q10D")
                        ],
                    },
                )
            self.assertEqual(manual.status_code, 200, manual.text)
            self.assertEqual(manual.json()["answers_submitted"], 4)
            self.assertEqual(manual.json()["answers_evaluated_successfully"], 4)
            self.assertEqual(len(manual.json()["submissions"]), 4)
            self.assertEqual(manual.json()["total_score"], 20.0)
            self.assertTrue(all(item["ai_score"] <= item["max_marks"] for item in manual.json()["submissions"]))

            def fail_one_answer(**kwargs):
                if kwargs["question_text"] == "Explain Q2B":
                    raise RuntimeError("mock grader failure")
                return _mock_evaluation()

            with patch.object(main.engine, "evaluate_answer", side_effect=fail_one_answer) as failing_evaluate:
                failed_job_response = self.client.post(
                    "/api/exams/MULTI_TEST/grade",
                    data={"student_id": "STUDENT4"},
                    files=[("files", ("answers.pdf", self._pdf_bytes(), "application/pdf"))],
                )
                self.assertEqual(failed_job_response.status_code, 200, failed_job_response.text)
                failed_job = self.client.get(f"/api/jobs/{failed_job_response.json()['job_id']}").json()
            self.assertEqual(failing_evaluate.call_count, 4)
            self.assertEqual(failed_job["diagnostics"]["answers_passed_to_grader"], 4)
            self.assertEqual(failed_job["diagnostics"]["answers_evaluated_successfully"], 3)
            self.assertTrue(failed_job["needs_review"])
            self.assertTrue(any("Grading failed" in item["reason"] for item in failed_job["diagnostics"]["skipped"]))

    def test_manual_grading_rejects_empty_answers(self):
        response = self.client.post(
            "/api/exams/BERT_TEST/grade-manual",
            json={
                "exam_id": "BERT_TEST",
                "student_id": "STUDENT3",
                "answers": [{"question_id": "Q1A", "answer": "   "}],
            },
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
