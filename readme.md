# Handwriting OCR and BERT Answer Grading

This application digitizes handwritten answer booklets and grades answers against teacher-authored reference answers and weighted rubrics.

## Pipeline

1. Accept PDF, PNG, JPG, or JPEG uploads.
2. Convert PDF pages to PNG with PyMuPDF, preserving upload and page order.
3. Preprocess each page with OpenCV and segment the handwriting into line images.
4. Read the line images with `microsoft/trocr-base-handwritten`.
5. Save the complete transcript for teacher review and split it into question answers.
6. Grade each answer with the BERT encoder `sentence-transformers/bert-base-nli-mean-tokens`, using the rubric, optional teacher reference answer, semantic similarity, and keyword coverage.
7. Show marks, similarity, coverage, and feedback in Batch Grading and Review & Calibrate.

OCR and BERT weights load lazily on the first corresponding request and are cached by Hugging Face. They are not downloaded during application startup. TrOCR is unloaded before BERT grading, and both models are released after a grading job. CUDA is used when available; CPU execution works without CUDA.

Question labels in the booklet (for example `Q1A`, `Q1B`, `Q4C`, or `Q10D`) improve automatic answer segmentation. The segmenter supports multi-digit question numbers and lettered parts, and keeps sections that do not match an exam rubric visible instead of dropping them. Grading reports OCR line, detected-section, mapped, submitted, successful, and skipped counts. When a section cannot be mapped or evaluated, the OCR transcript and segmented text are available in Batch Grading for manual editing, question assignment, and regrading. Manual submission is only finalized after every non-empty submitted answer has been evaluated.

## Install and run on Windows PowerShell

Use Python 3.11 or 3.12. Open two PowerShell terminals in the project directory.

### Backend

```powershell
cd .\backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks virtual-environment activation, run the backend with the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

PyTorch can use a compatible NVIDIA CUDA runtime when one is installed. It falls back to CPU if CUDA is unavailable or model placement runs out of GPU memory. CUDA is not required.

### Frontend

```powershell
cd .\frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The backend API documentation is available at `http://127.0.0.1:8000/docs`.

## Models and grading

- OCR: `microsoft/trocr-base-handwritten`
- Grading: `sentence-transformers/bert-base-nli-mean-tokens` loaded through Transformers as a BERT encoder
- Each question's maximum score is enforced by its rubric weights.
- A teacher reference answer is optional; when omitted, the rubric criteria are used as the semantic reference.
- Review & Calibrate retains teacher score verification and rubric-weight calibration.

## Tests

From the project directory:

```powershell
.\backend\.venv\Scripts\python.exe -m unittest discover -s .\backend -p "test_*.py" -v
```

The tests cover PDF page conversion, OpenCV line segmentation, bounded BERT rubric scoring, reference-answer persistence, and the batch and individual grading APIs. Model downloads and full OCR inference are deliberately excluded from the automated suite.

## Batch-grade an existing SQLite database

Activate the project's Python environment, then run this from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_all_students_bert.py
```

By default, the script reads `~/Downloads/grading_system.db` (or falls back to `backend/grading_system.db`) and writes `answer1.csv` plus `bert_evaluation_log.csv` in the project root. To select a different database or output folder, pass `--db PATH` or `--output-dir PATH`; pass `--exam-id ID` when the database contains more than one exam.

The script opens the database read-only and reuses the existing `sentence-transformers/bert-base-nli-mean-tokens` grader and question segmenter. Question-level submission rows take precedence over booklet transcripts. A booklet is segmented only when no question-level submissions exist for that student; unmarked transcripts are logged for review rather than assigned to an arbitrary question. When a rubric has no explicit reference answer, the grader uses the rubric's key points as its reference, as in the existing scoring implementation. Scores are BERT predictions; `Verified Sum` separately totals existing teacher scores marked verified and does not affect predicted marks or IA totals.

The ledger follows the project's IA rule: `max(Q1A + Q1B, Q2A + Q2B) + (Q3A + Q3B)`, out of 20. Unattempted questions count as zero; failed or ambiguous evaluations remain blank and require review. The log includes both evaluated answers and expected questions with no submission. Submission rows belonging to an unregistered/other exam with no corresponding rubric are logged as skipped, not scored using this exam's rubric.
