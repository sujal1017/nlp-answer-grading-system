# AI Grader - Automated Handwritten Evaluation System

A local optical character recognition and evaluation pipeline that processes handwritten test papers into clean, structured Markdown using the open-weight Qwen2-VL model.

---

## 1. Project Overview and Working Mechanism

1. **Input**: Scanned or photographed images of handwritten exam sheets are placed in the `data/` folder.
2. **Vision-Language Processing**: The local `Qwen/Qwen2-VL-2B-Instruct` model ingests the image directly and parses handwriting, layout headers, dates, and bulleted lists.
3. **Markdown Normalization**: The transcription is written to a structured `.md` file, which can be used as standardized input for later rubric grading steps.
4. **Local Hardware Execution**: The pipeline runs on the host machine with PyTorch and CUDA acceleration, avoiding external API costs, transmission latency, and service rate limits.

---

## 2. Directory Structure

```text
ai-grader/
|
|-- data/
|   |-- answer.jpeg
|   |-- answer2.jpeg
|   |-- answer3.jpeg
|
|-- ocr.py                 # Core extraction and transcription script
|-- qwen_ocr_output.md     # Generated Markdown output
`-- readme.md              # Project documentation
```

---

## 3. Environment and Hardware Prerequisites

- **Operating System**: Windows 10/11 or a modern Linux distribution
- **Python**: 3.10-3.13
- **Hardware**: Dedicated NVIDIA GPU with at least 6 GB VRAM, such as an RTX 3050, RTX 4050, or higher, for accelerated inference

---

## 4. Step-by-Step Installation

### Step A: Clone the Repository and Configure a Virtual Environment

Open your terminal in the target workspace and set up an isolated environment:

```bash
git clone <your-repository-url>
cd ai-grader
python -m venv venv
```

Activate the environment.

On Windows CMD or PowerShell:

```powershell
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### Step B: Install PyTorch with CUDA Support

A standard `pip install torch` may install CPU-only binaries. Use the dedicated wheel index that matches your NVIDIA driver and CUDA toolchain. For CUDA 12.6:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verify that PyTorch can detect your dedicated GPU:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

Expected output:

```text
CUDA Available: True
```

If the command returns `False`, verify your NVIDIA drivers and reinstall the CUDA-enabled PyTorch build.

### Step C: Install Model Dependencies

Install the libraries needed for multimodal input formatting and model loading:

```bash
pip install transformers accelerate torchvision qwen-vl-utils
```

---

## 5. Running the Pipeline

1. Place the target test image inside the `data/` directory.
2. Update `image_file` in `ocr.py` if you want to process a different file. The current script uses `data/answer4.jpeg`.
3. Run the main processing script:

   ```bash
   python ocr.py
   ```

4. On the first run, Hugging Face downloads the `Qwen2-VL-2B-Instruct` checkpoint, which is roughly 4.5 GB, into your local cache directory. Later runs load the model from disk.
5. When processing finishes, inspect `qwen_ocr_output.md` to review the extracted Markdown text.

---

## 6. Troubleshooting Common Issues

- **`torch.cuda.is_available()` returns `False`**: You may be using a CPU-only PyTorch build. Run `pip uninstall torch torchvision -y`, then reinstall PyTorch using the CUDA wheel index shown in Step B.
- **Process defaults to Intel or integrated GPU**: Ensure the script uses `device_map="auto"` and set your operating system's graphics preference for the Python executable to high-performance NVIDIA graphics.
- **Out-of-memory errors**: If high-resolution images exhaust VRAM, downscale the input image or reduce `max_new_tokens` in the generation call inside `ocr.py`.
