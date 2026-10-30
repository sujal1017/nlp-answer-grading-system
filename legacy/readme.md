# AI Grader --- Automated Handwritten Answer Evaluation

A local, AI-assisted system for processing handwritten examination
papers, preparing marking rubrics, evaluating student answers, and
allowing teachers to review suggested marks.

The application uses a Streamlit web interface and the Qwen2-VL
vision-language model. It is intended to assist teachers, not replace
their final judgement. OCR and AI grading can make mistakes, so results
should be reviewed before marks are finalized.

## Features

-   **Question-paper and rubric workflow:** prepare weighted marking
    criteria.
-   **Handwritten booklet processing:** process images of student answer
    pages using a vision-language model.
-   **Answer parsing:** split extracted text into question-level
    answers.
-   **Rubric-based grading:** generate suggested marks and feedback
    against configured criteria.
-   **Teacher verification:** review and correct suggested results.
-   **Local database:** store exam, rubric, submission, and evaluation
    records in SQLite.
-   **Feedback-based calibration:** use verified examples and
    rubric-weight updates where supported by the current application
    workflow.

> Grading output is a recommendation. Teachers should verify
> transcription and scores before using them as official results.

## Project structure

``` text
nlp-answer-grading-system/
├── app.py              # Streamlit user interface and workflow
├── engine.py           # Qwen2-VL loading, transcription, and grading logic
├── chunker.py          # Question-paper and answer-booklet parsing
├── database.py         # SQLite database operations
├── requirements.txt    # Python package dependencies
├── readme.md           # Project documentation
└── .gitignore
```

The application may create a local SQLite database and other runtime
files. Keep generated databases, uploaded student papers, model caches,
and secrets out of public version control unless intentionally shared.

## Requirements

-   Windows 10/11 or a compatible Linux environment
-   Python **3.10--3.13** (recommended by the project's environment
    notes)
-   NVIDIA GPU and a compatible NVIDIA driver for CUDA acceleration
-   Internet connection on the first run to download model weights
-   Sufficient RAM, disk space, and GPU memory

Qwen2-VL-2B can require several gigabytes of storage and memory. A GPU
with 6 GB or more VRAM is preferable. A GPU with 4 GB VRAM may require
CPU/disk offloading, run slowly, or encounter out-of-memory errors.

## Setup on Windows PowerShell

### 1. Clone the repository

``` powershell
git clone https://github.com/sujal1017/nlp-answer-grading-system.git
cd nlp-answer-grading-system
```

If you already have the repository locally, open PowerShell in that
repository folder instead.

### 2. Create and activate a virtual environment

For example, if Python 3.12 is installed:

``` powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

Use an installed Python version between 3.10 and 3.13. If PowerShell
blocks activation, run this in the current terminal session:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 3. Install CUDA-enabled PyTorch

For a compatible NVIDIA driver, install the CUDA 12.6 wheel build:

``` powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

If your system requires a different PyTorch/CUDA wheel, use the official
PyTorch installation selector and choose the appropriate operating
system, package manager, Python, and compute platform.

Verify the installation:

``` powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"
```

`CUDA available: True` indicates that PyTorch can access CUDA. If it
reports `False`, check the NVIDIA driver and installed PyTorch build.

### 4. Install project dependencies

``` powershell
python -m pip install -r requirements.txt
```

PyTorch is installed separately because the appropriate wheel depends on
the user's hardware and CUDA setup.

### 5. Launch the web application

From the repository root, with the virtual environment active:

``` powershell
python -m streamlit run app.py
```

Streamlit prints a local URL, typically `http://localhost:8501`. Open
that URL in your browser.

## Typical workflow

1.  Open the Streamlit application.
2.  Create or select an exam.
3.  Provide the question paper and prepare the marking rubric.
4.  Upload the student's answer-booklet pages.
5.  Let the application transcribe and parse the answers.
6.  Review the suggested marks and feedback.
7.  Correct or approve the evaluation before treating it as final.

The exact input fields and order may vary with the current UI
implementation.

## Troubleshooting

### `ModuleNotFoundError: No module named 'streamlit'`

Make sure the virtual environment is active, then run:

``` powershell
python -m pip install streamlit
```

### Another package is missing

Check that the correct virtual environment is active, then run:

``` powershell
python -m pip install -r requirements.txt
```

If a missing module is not declared in `requirements.txt`, confirm that
the code needs it and add it to the file.

### CUDA is unavailable

Check the NVIDIA driver and verify that the CUDA-enabled PyTorch build
is installed in the same Python environment used to launch Streamlit.

### GPU memory errors or slow inference

Close other GPU-heavy applications, reduce image resolution if supported
by the workflow, and review the model's device/offloading settings in
`engine.py`. CPU offloading may reduce GPU memory usage but can make
inference substantially slower.

### Model downloads on first run

The model weights are downloaded from Hugging Face when first needed and
cached locally. Ensure you have an internet connection and enough disk
space.

### The application starts but evaluation fails

Read the full terminal traceback. Confirm that the expected Python
modules are present and that uploaded files match the formats supported
by the current code.

## Dependencies

-   **Streamlit:** browser-based application interface.
-   **PyTorch:** model execution and GPU computation (installed
    separately).
-   **Transformers:** loads the Qwen2-VL model and processor.
-   **Accelerate:** supports automatic device placement and offloading.
-   **Qwen-VL-Utils:** prepares image inputs for Qwen vision-language
    processing.
-   **Torchvision:** supporting image/vision dependency; retained
    pending compatibility testing.

Python standard-library modules such as `os`, `json`, `re`, and
`sqlite3` do not need to be installed separately.

## Responsible use and evaluation

-   Verify extracted text when handwriting is unclear.
-   Keep student examination data private and restrict access to
    uploaded papers and stored results.
-   Test the system on representative, teacher-marked answers before
    relying on grading outputs.
-   Track grading quality using an agreed evaluation set and metrics
    such as mean absolute error and agreement with teacher marks.
-   Treat model-generated marks and feedback as suggestions until a
    teacher has reviewed them.

## Development note

Keep this README aligned with the actual source code and tested
dependency versions. Before a release, test installation in a fresh
virtual environment and record the Python, PyTorch, Transformers, and
model versions that work together.
