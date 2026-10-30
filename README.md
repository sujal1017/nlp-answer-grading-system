# AI-Powered Handwritten Answer Sheet Evaluation System

A Flask + React application that takes an exam question paper and a student's handwritten answer sheet (both PDF), extracts the text, maps answers to questions, grades them against an evaluator-defined rubric, and produces explained marks, human-review tools and PDF/CSV reports.

**AI marks are suggestions.** A human reviewer approves every final mark. The application does not replace examiners, and reports label AI-generated and human-approved marks separately.

## Layout

| Path | What |
|---|---|
| `backend/` | Flask API (`app/routes`), SQLAlchemy models, PDF/OCR/mapping services, grading engine (`app/grading`), reports, migrations, tests |
| `frontend/` | React + Vite + TypeScript + Tailwind UI (talks only to the real API, no mock data) |
| `legacy/` | The original Streamlit prototype, kept for reference. Not used. |

## Requirements

- Python 3.13 (tested), Node 20+
- **Tesseract binary**: `apt install tesseract-ocr` (macOS: `brew install tesseract`)
- No GPU needed. The app starts and works on CPU with only the packages in `backend/requirements.txt`.

## Setup and run

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python wsgi.py                 # http://127.0.0.1:5000  (applies DB migrations automatically)

# frontend (second terminal)
cd frontend
npm install
npm run dev                    # http://localhost:5173  (proxies /api to :5000)
```

Production build of the UI: `npm run build` (output in `frontend/dist`).

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | SQLite in `backend/instance/app.db` | Use `postgresql+psycopg://user:pass@host/db` for PostgreSQL (`pip install "psycopg[binary]"`) |
| `UPLOAD_DIR` | `backend/uploads` | Stored PDFs |
| `MAX_UPLOAD_MB` | 25 | Upload size limit |
| `SECRET_KEY` | dev value | **Change in production** |
| `CORS_ORIGINS` | `http://localhost:5173` | |
| `AUTO_MIGRATE` | 1 | Set 0 to run `flask --app wsgi db upgrade` manually |
| `EXTERNAL_LLM_URL` / `_KEY` / `_MODEL` | empty | Optional OpenAI-compatible service for AI judging / drafting reference answers. **Page text is sent to that service**; costs and privacy are yours to assess. Keep the key out of git. |

Runtime options (OCR engine, DPI, embedder, judge, page limit, retention, concept credits) are editable on the Settings page.

## Workflow in the UI

1. **Examinations → New**: create an exam, upload the question-paper PDF. Questions and marks are extracted; anything uncertain is flagged for confirmation. Edit, add or delete questions.
2. For each question set the **reference answer** and **rubric** (criteria with max marks and concepts). Reference answers drafted by AI are labelled and must be approved before grading.
3. Upload a student's **answer sheet PDF**. Each page is read directly (selectable text) or via OCR (scanned).
4. **Pages & OCR**: compare the page image with the text, correct it, mark pages reviewed.
5. **Answer mapping**: confirm which text belongs to which question. Unanswered, unreadable and ambiguous are distinct states; nothing is silently guessed.
6. **Evaluation**: run grading. Each answer shows criterion marks, concept status (correct / partial / mentioned / incorrect / missing) with evidence, reasons, improvements and a stronger-answer outline. Override marks per criterion, comment, and approve.
7. **Results**: totals, per-question marks, strengths/weaknesses, PDF and CSV download. History lists saved evaluations.

If grading cannot be done reliably (no rubric, no approved reference, judge unavailable, invalid AI output) the answer is routed to **manual grading**. No marks are fabricated.

## Grading design

The engine (`backend/app/grading`) scores by **rubric criteria and concepts**, not by whole-answer similarity. Embedding similarity is only one input to judging each concept and never sets marks alone. Final marks are computed and clamped by backend validation (criterion ≤ max, question ≤ max, no marks for length, no deductions for spelling). Original OCR text, corrected text, AI scores, human scores and final approved scores are stored separately. The default judge is a local heuristic; an optional external LLM judge uses a validated structured schema with limited retries.

AI-vs-human metrics (MAE, RMSE, Pearson, Spearman, exact agreement, within-tolerance) are available at `GET /api/exams/<id>/metrics` and are withheld below a minimum sample size.

## Tests

```bash
cd backend && python -m pytest -q
```

Current result on the development machine: **101 passed, 1 skipped**.

What is real vs mocked:
- `tests/test_ocr_real.py` runs the real Tesseract binary on synthetic printed images and three real handwriting photos (`tests/fixtures/handwriting/`).
- `tests/test_api_e2e.py` runs the full API workflow (upload → extract → rubric → OCR → correct → map → evaluate → override → export → retrieve), with **scripted (mocked) OCR text** for the scanned pages and the real heuristic judge. Jobs run inline in tests, so thread-related bugs are only covered by the manual live run below.
- The skipped test needs `sentence-transformers` (not installed in the test environment), so **semantic paraphrase quality is untested**. Without it the lexical fallback embedder is used, which is not semantic.
- **Qwen2-VL OCR is not tested** (needs torch + a model download). Do not assume it works on your hardware until you try it.

Also verified manually against a live server: create exam, upload question paper, upload a handwritten PDF, review/correct OCR, map, evaluate, human override and approve, download PDF/CSV, restart the server and retrieve the saved results.

## Known limitations

- **Tesseract is poor at handwriting.** On the sample handwriting photos it returned mostly garbled text (confidence about 50). The app flags this and relies on human correction. For real use, configure a handwriting-capable engine (Qwen2-VL locally, or `remote_vlm`), and measure its accuracy on your own sheets first.
- The default heuristic judge is conservative and keyword/similarity based; expect to review marks. It has not been validated against a human-graded dataset.
- No authentication/authorization is implemented (single trusted user assumed).
- Background jobs are threads in the API process; fine for one server, not a distributed queue.

## Troubleshooting

- *"Cannot reach the backend"* in the UI: start `python wsgi.py` in `backend/` (port 5000).
- *OCR unavailable*: install the `tesseract` binary and ensure it is on `PATH`.
- *Page orientation warnings*: orientation detection needs Tesseract OSD data (`tesseract-ocr-osd`); pages are assumed upright otherwise.
- *Large uploads rejected*: raise `MAX_UPLOAD_MB`.
