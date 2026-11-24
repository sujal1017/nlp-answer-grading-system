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

Question labels in the booklet (for example `Q1A`, `Q1B`) improve automatic answer segmentation. If no sections can be matched, the OCR transcript remains saved and is available for manual review and assignment.

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
