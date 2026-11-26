# RubrixAI — Automated Handwritten Exam Evaluation Platform

EvalAI is an end-to-end, offline-capable handwritten exam evaluation system tailored for university examinations and internal assessments (IA). It combines local Vision-Language Models (Qwen2-VL), reference textbook Retrieval-Augmented Generation (RAG), multi-tier semantic answer sheet chunking, anti-stuffing cross-encoder verification, and institutional choice-rule grading into a unified full-stack application.

---

## 1. System Overview & Key Capabilities

- **Offline Handwritten Answer Digitization**:
  - Employs `Qwen/Qwen2-VL-2B-Instruct` locally with PyTorch CUDA acceleration for verbatim extraction of handwritten text.
  - **Dynamic Top Margin & Ink Bounding-Box Cropping**: Strips header metadata (top 12%) and masks margins to identify actual ink regions (`arr < 155`), physically trimming blank lower halves (`max_y + 50`) and dynamically scaling token limits (`max_tokens = max(160, int(1024 * (cropped_h / orig_h)))`).
  - **3-Stage OCR Fallback Guarantee**: Full-page greedy decoding -> Horizontal top/bottom half split pass (triggered if word count < 15) -> Mild sampling fallback (`temperature=0.2`).
  - **OCR Sanitizer & Anti-Looping Guard**: Enforces `no_repeat_ngram_size=4` and `repetition_penalty=1.18`. Strips visual grounding tags (`<|box_start|>`, `<box>`), bounding coordinates `(x, y)`, conversational filler, code fences, runaway number sequences (8+ digits), and multi-word phrase repetition loops.
  - **Margin Marker Extraction**: Crops the left 18% margin strip to reliably capture handwritten question markers (`2] a`, `3b)`).

- **Reference Textbook RAG Grounding**:
  - Ingests prescribed reference textbook / lecture notes PDFs per exam on CPU (zero GPU VRAM overhead).
  - Uses PyMuPDF (`fitz`) to extract text and splits it into ~220-word overlapping chunks tagged with page numbers.
  - Encodes passages using singleton `sentence-transformers/all-MiniLM-L6-v2` and persists normalized embeddings to `backend/data/textbooks/{exam_id}_index.npz`.
  - Performs cosine similarity retrieval to ground rubric criteria and blend student answer scores with textbook definitions (`0.75 * S_sem + 0.25 * S_book`).

- **Multi-Tier Semantic Boundary Chunker**:
  - Converts multi-page student answer booklet PDFs into high-resolution images.
  - Normalizes varied handwritten margin markers (`normalize_margin_markers`).
  - Protects Roman numerals (`(i)-(iv)`) and internal lists from triggering false segment breaks.
  - Detects multi-page answer continuations.
  - **Semantic Orphan Routing**: Employs vector similarity to route orphaned subquestions (`a)`, `b)`) to the most appropriate parent question (`Q1`, `Q2`, `Q3`).

- **EduEval Anti-Stuffing & Cross-Encoder Scoring**:
  - **Heading-Echo Filter**: Detects and strips standalone lines that copy question text or rubric headings (`token_set_ratio >= 82`), blocking marks for heading parroting.
  - **Global Relevance Gate**: Computes semantic similarity against the question prompt; scales down scores if overall similarity is below `0.30`.
  - **Cross-Encoder Entailment Verification**: Runs `cross-encoder/ms-marco-MiniLM-L6-v2` (on CPU) across student answer windows against rubric criteria (requires sigmoid probability `>= 0.35` for a point to be marked as **Matched**).

- **SDMCET Internal Assessment (IA) Choice Rules**:
  - Evaluates internal assessment structure out of 20 marks:
    - **Module 1**: Best of Question 1 (`Q1A + Q1B`) OR Question 2 (`Q2A + Q2B`) [Max 10 Marks].
    - **Module 2**: Question 3 (`Q3A + Q3B`) compulsory [Max 10 Marks].
  - Computes final IA score (`final_ia_total`) automatically across student booklets.

- **Persistent Scorecards & Curation Studio**:
  - SQLite persistence (`backend/grading_system.db`) tracking exams, rubrics, raw booklets, student answers, and persistent student scorecards (`student_results`).
  - Review & curation tools: side-by-side original page inspection, segment reassignment, cursor-based text splitting (`split_at_cursor`), and segment merging.
  - Continuous teacher override logging (`training_dataset.jsonl`) for weight calibration.
  - Comprehensive **Marks Ledger** with live filtering, evaluation status tracking, and one-click CSV / Excel export.

---

## 2. Directory Structure

```text
ai-grader/
|-- backend/
|   |-- data/                     # Exams, raw booklets, textbook indexes, training dataset
|   |   |-- booklets/             # Uploaded student booklet PDF pages
|   |   |-- exams/                # Exam metadata & parsed questions
|   |   |-- textbooks/            # Indexed reference textbook NPZ files
|   |   `-- training_dataset.jsonl # Feedback logs for calibration
|   |-- chunker.py                # Booklet PDF splitting, margin normalization, IA choice rules
|   |-- database.py               # SQLite schema, student_results sync, booklet persistence
|   |-- engine.py                 # Qwen2-VL OCR, ink region cropping, rubric generation, sanitization
|   |-- evaluate_accuracy.py      # Accuracy benchmark evaluation script
|   |-- grading_system.db         # SQLite database file
|   |-- main.py                   # FastAPI REST API endpoints
|   |-- rag.py                    # PyMuPDF textbook ingestion, MiniLM CPU vector indexing & retrieval
|   |-- requirements.txt          # Backend Python dependencies
|   |-- scorer.py                 # EduEval deterministic scorer, heading echo filter, cross-encoder
|   `-- test_pipeline.py          # 26 automated unit and end-to-end integration tests
|
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |   |-- BatchGrading.jsx   # Multi-booklet upload, OCR progress, batch evaluation
|   |   |   |-- KpiCards.jsx       # Real-time metrics (graded, pending, average IA score)
|   |   |   |-- QuestionStudio.jsx # Question parsing, Bloom's rubrics, textbook PDF upload
|   |   |   |-- ResultsLedger.jsx  # Student scorecard ledger with CSV/Excel export
|   |   |   `-- ReviewGrade.jsx    # Split-screen booklet viewer, segment curation, score override
|   |   |-- services/              # Axios API service bindings
|   |   |-- App.jsx                # Tab navigation & telemetry header
|   |   |-- main.jsx               # React entry point
|   |   `-- index.css              # Tailwind CSS styling
|   |-- package.json               # Frontend dependencies & Vite scripts
|   |-- tailwind.config.js         # Tailwind configuration
|   `-- vite.config.js             # Vite development server configuration
|
|-- test/                         # Sample student answer booklets (test.pdf - test36.pdf)
|-- test_pipeline.py              # Root test runner shortcut
`-- README.md                     # Project documentation
```

---

## 3. Hardware & Environment Prerequisites

- **Operating System**: Windows 10/11 or modern Linux distribution
- **Python**: 3.10 – 3.13
- **Node.js**: 18.x or later
- **GPU (Recommended)**: NVIDIA GPU with at least 6 GB VRAM (e.g., RTX 3050, RTX 4050, RTX 4060, or higher) with CUDA 12.x support.
  - *Note*: Embedding generation (MiniLM) and Cross-Encoder entailment run entirely on CPU to reserve VRAM exclusively for Qwen2-VL. CPU-only execution is supported as a fallback.

---

## 4. Installation & Setup

### Step A: Clone the Repository & Configure Python Virtual Environment

```bash
git clone <your-repository-url>
cd ai-grader
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### Step B: Install PyTorch with CUDA Acceleration

For CUDA 12.6 support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU detection:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### Step C: Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step D: Install Frontend Dependencies

Open a new terminal window:

```bash
cd frontend
npm install
```

---

## 5. Running the Application

### 1. Launch the Backend API

From the project root:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI Swagger documentation is accessible at `http://localhost:8000/docs`.

### 2. Launch the Frontend UI

In a separate terminal:

```bash
cd frontend
npm run dev
```

Open your browser at `http://localhost:5173` to access the EvalAI web interface.

---

## 6. End-to-End Workflow

```
+-----------------------------------------------------------------------------+
| 01. QUESTION STUDIO                                                         |
| Paste questions -> Generate Bloom's rubrics -> Upload reference textbook PDF|
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 02. BATCH GRADING                                                           |
| Upload student PDFs -> Qwen2-VL OCR -> Multi-tier Chunker -> EduEval Scorer |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 03. REVIEW & CALIBRATE                                                      |
| Inspect page images -> Reassign / Split / Merge segments -> Override marks |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| 04. MARKS LEDGER                                                            |
| View sub-question breakdown -> Apply SDMCET Choice Rules -> Export CSV/Excel|
+-----------------------------------------------------------------------------+
```

1. **Question Studio (`01. Question Studio`)**:
   - Create an Exam ID (e.g., `IA1_RM_2026`).
   - Paste the exam question paper (supports institutional formats like `Q1 (a)`, `Q1 (b)`, `Q2 (a)`, `Q2 (b)`, `Q3 (a)`, `Q3 (b)`).
   - Generate Bloom's Taxonomy rubric criteria with calibrated weights.
   - Upload the reference textbook / notes PDF to build an indexed RAG knowledge base.

2. **Batch Grading (`02. Batch Grading`)**:
   - Drag and drop student handwritten booklet PDFs (e.g., from `test/test.pdf`).
   - The engine renders pages, applies top margin and dynamic ink bounding-box cropping, performs 3-stage fallback OCR, normalizes markers, and segments answers.
   - The EduEval engine grades each answer against the rubric points with anti-stuffing cross-encoder checks and textbook grounding.

3. **Review & Calibrate (`03. Review & Calibrate`)**:
   - Inspect digitized text alongside original booklet pages.
   - Use in-line curation tools to reassign answers, split merged answers at cursor positions, or merge adjacent segments.
   - Review rubric criterion matches and override marks with feedback; overrides update the training dataset for future weight calibration.

4. **Marks Ledger (`04. Marks Ledger`)**:
   - Review each student's scorecard: individual question marks (`Q1A`, `Q1B`, `Q2A`, `Q2B`, `Q3A`, `Q3B`), pair totals (`Q1 Total`, `Q2 Total`, `Q3 Total`), and the final IA score out of 20 (`Best(Q1, Q2) + Q3`).
   - Download the full score ledger as a formatted CSV or Excel spreadsheet.

---

## 7. Running the Automated Test Suite

The repository includes an extensive automated test suite covering chunking, OCR sanitization, database persistence, anti-stuffing guards, RAG retrieval, and choice rules.

Run the test suite from the project root:

```bash
python test_pipeline.py
```

Or directly inside the `backend/` directory:

```bash
python backend/test_pipeline.py
```

### Verified Test Cases:
1. `test_margin_normalization`: Normalizes varied OCR marker formats (`2] (6)`, `36)`, `3] a`) into standard `Q2B)`, `Q3B)`, `Q3A)`.
2. `test_hybrid_chunker_does_not_collapse_to_q1a`: Validates segmentation across questions without collapsing to the first key.
3. `test_orphan_b_inherits_q1`: Verifies that an unlabelled `b)` follows the preceding `Q1A` into `Q1B`.
4. `test_multipage_continuation_chunking`: Handles answers spanning across multiple page boundaries.
5. `test_multi_tier_margin_segmentation`: Handles split markers across lines.
6. `test_booklet_chunking_test_pdf_and_test1_pdf`: Verifies chunking on realistic multi-page PDF booklets.
7. `test_roman_numerals_and_internal_lists_no_split`: Ensures lists `(i)-(iv)` do not trigger false question splits.
8. `test_orphan_subquestion_semantic_routing`: Embeds orphan subquestions and routes them using semantic cosine similarity.
9. `test_raw_booklet_persistence`: Verifies full verbatim booklet persistence and retrieval.
10. `test_assign_segment_with_index_0_and_all_questions`: Reassigns segments and updates live IA totals.
11. `test_split_at_cursor_index_0_and_positive`: Validates cursor-based answer splitting at index 0 and mid-text.
12. `test_delete_student_submission`: Deletes submissions and recalculates IA totals cleanly.
13. `test_merge_and_reassign_and_split`: Verifies atomic segment merging, reassignment, and re-splitting.
14. `test_dump_q1a_split_char0_and_evaluate_workflow`: End-to-end test of raw booklet dumping, cursor splitting, and grading.
15. `test_generation_kwargs_clean_sampling`: Ensures sampling flags (`temperature`, `top_p`) are cleanly excluded during greedy decoding.
16. `test_sanitize_hallucinations_and_coordinate_stripping`: Strips bounding boxes, visual tags, and synthetic quizzes.
17. `test_strip_chat_prefixes_and_runaway_numbers`: Strips conversational preambles and runaway number sequences (8+ standalone digits).
18. `test_strip_repetition_loops_and_crop_to_ink`: Verifies truncation of 3+ repeated multi-word phrase loops and ink bounding-box cropping.
19. `test_preprocess_page_image_crops_top_margin`: Validates cropping of top header margin (12%).
20. `test_choice_rule`: Tests SDMCET IA choice rule logic (`Best(Q1, Q2) + Q3`).
21. `test_teacher_parse_and_rubric`: Tests teacher question parsing and Bloom's atomic rubric generation.
22. `test_database_and_calibration`: Verifies SQLite schema creation and live weight calibration.
23. `test_transcribe_images_fallback_retry_recovers_page`: Simulates low word count in Attempt 1 and verifies recovery in Attempt 2 (split halves).
24. `test_edueval_deterministic_scoring_q1a`: Validates rubric matching and mark calculation on a benchmark answer.
25. `test_anti_stuffing_guard_and_heading_parroting`: Rejects copied question headings and filler text via Cross-Encoder entailment verification.
26. `test_textbook_rag_indexing_and_retrieval`: Ingests a reference PDF, generates CPU MiniLM embeddings, and retrieves top context chunks.
27. `test_persistent_student_scorecard_sync`: Verifies persistent synchronization of student scorecards and teacher overrides.

---

## 8. Technology Stack

- **Backend**:
  - [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/): Asynchronous REST API
  - [PyTorch](https://pytorch.org/) & [Hugging Face Transformers](https://huggingface.co/docs/transformers): `Qwen/Qwen2-VL-2B-Instruct` multimodal inference
  - [Sentence-Transformers](https://www.sbert.net/): `all-MiniLM-L6-v2` (embeddings) & `ms-marco-MiniLM-L6-v2` (Cross-Encoder)
  - [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/): High-resolution PDF rendering and textbook text extraction
  - [Pillow (PIL)](https://python-pillow.org/): Image preprocessing, autocontrast, sharpening, ink region cropping
  - [SQLite3](https://sqlite.org/): Relational database storage
  - [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz): Fast string matching and heading-echo filtering

- **Frontend**:
  - [React 18](https://react.dev/) & [Vite](https://vitejs.dev/): Modern single-page application framework
  - [Tailwind CSS](https://tailwindcss.com/): Responsive UI design
  - [Lucide React](https://lucide.dev/): Icon set
  - [Axios](https://axios-http.com/): HTTP client for API communication
