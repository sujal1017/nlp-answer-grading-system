import json
import os
import re
import threading
from typing import Dict, List, Optional, Union

# Enforce offline huggingface operation
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
torch = None
TORCH_AVAILABLE = False


def _ensure_torch():
    global torch, TORCH_AVAILABLE
    if TORCH_AVAILABLE and torch is not None:
        return
    import torch as _torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    torch = _torch
    globals()["AutoProcessor"] = AutoProcessor
    globals()["Qwen2VLForConditionalGeneration"] = Qwen2VLForConditionalGeneration
    globals()["process_vision_info"] = process_vision_info
    TORCH_AVAILABLE = True


BLOOM_TEMPLATES = {
    "define": [
        "Accurate definition and core meaning of {topic}",
        "Essential characteristics / distinguishing features of {topic}",
        "Relevant classification, types, or related terms",
        "Example or academic significance of {topic}",
    ],
    "explain": [
        "Clear explanation of the meaning of {topic}",
        "Underlying process, reasons, or working of {topic}",
        "Key factors, variables, or components involved",
        "Implications, outcome, or academic relevance",
    ],
    "describe": [
        "Systematic description of {topic}",
        "Ordered steps, stages, or procedure",
        "Purpose of each major step",
        "Expected outcome or use in research practice",
    ],
    "discuss": [
        "Concept and scope of {topic}",
        "Arguments, viewpoints, or theoretical basis",
        "Applications or illustrations in research",
        "Limitations, issues, or concluding remark",
    ],
    "compare": [
        "Identification of items being compared in {topic}",
        "Similarities with supporting points",
        "Differences with supporting points",
        "Justified conclusion or preference",
    ],
    "differentiate": [
        "Clear identification of the two sides in {topic}",
        "Contrasting attributes presented in parallel",
        "Examples that make the distinction concrete",
        "Summary of why the distinction matters",
    ],
    "list": [
        "Coverage of the required items in {topic}",
        "Correct naming / terminology of each item",
        "Brief elaboration of the listed points",
        "Organisation and completeness of the list",
    ],
    "evaluate": [
        "Criteria used to evaluate {topic}",
        "Balanced strengths and weaknesses",
        "Evidence or examples supporting the judgement",
        "A reasoned concluding assessment",
    ],
}

VERB_ORDER = [
    "differentiate", "compare", "evaluate", "discuss", "describe",
    "explain", "define", "list", "state", "write",
]


def calibrate_weights(n: int, max_marks: float) -> List[float]:
    n = max(n, 1)
    base = round(float(max_marks) / n, 2)
    weights = [base] * n
    drift = round(float(max_marks) - sum(weights), 2)
    weights[0] = round(weights[0] + drift, 2)
    return weights


def generate_atomic_rubric(question_text: str, max_marks: float, num_points: int = 4) -> List[Dict]:
    """Deterministic atomic rubric (no VLM). Teacher can edit weights in the UI."""
    q = (question_text or "").strip()
    verb = "explain"
    for v in VERB_ORDER:
        if re.search(rf"\b{v}\b", q, re.I):
            verb = "list" if v in {"list", "state", "write"} else v
            if verb not in BLOOM_TEMPLATES:
                verb = "explain"
            break

    topic = re.sub(
        r"^\s*(briefly\s+)?(define|explain|describe|discuss|compare|differentiate|list|state|write|evaluate)\b[\s:,-]*",
        "",
        q,
        flags=re.I,
    ).strip(" .:?")
    if not topic:
        topic = q[:80] or "the asked concept"

    templates = BLOOM_TEMPLATES.get(verb, BLOOM_TEMPLATES["explain"])[:num_points]
    points = [t.format(topic=topic) for t in templates]
    weights = calibrate_weights(len(points), max_marks)
    return [{"point": pt, "weight": w} for pt, w in zip(points, weights)]


def collapse_word_loops(text: str) -> str:
    """Cut hallucinated phrase loops such as repeating the same 6–20 word span."""
    words = text.split()
    if len(words) < 12:
        return text
    for window in range(min(24, len(words) // 3), 5, -1):
        out: List[str] = []
        i = 0
        n = len(words)
        while i < n:
            chunk = words[i : i + window]
            if len(chunk) < window:
                out.extend(words[i:])
                break
            repeats = 1
            j = i + window
            while words[j : j + window] == chunk:
                repeats += 1
                j += window
            if repeats >= 3:
                out.extend(chunk * 2)
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def collapse_line_loops(text: str) -> str:
    result = []
    prev = None
    streak = 0
    for line in text.splitlines():
        key = re.sub(r"\s+", " ", line).strip()
        if key and key == prev:
            streak += 1
            if streak >= 2:
                continue
        else:
            streak = 0
            prev = key
        result.append(line)
    return "\n".join(result)


# Broad regex matching any synthetic quiz question patterns generated by small VLMs
QUIZ_CUTOFF_RE = re.compile(
    r"(?:^|\n)\s*(?:###?\s*)?(?:"
    r"Question\s+\d+\s*[:.-]\s*(?:What|Define|Explain|Describe|List|How|State|Discuss|Give|Write)\b|"
    r"(?:Follow-up|Study|Practice|Sample)\s+Questions?\b|"
    r"Quiz\s*[:\n])",
    re.IGNORECASE,
)

SIGN_OFF_RE = re.compile(
    r"(?:^|\n)\s*(?:I hope this helps|Let me know if|As an AI|Hope this (?:helps|transcription)|"
    r"If you (?:have|need) any|Feel free to ask)\b",
    re.IGNORECASE,
)

PREFIX_RE = re.compile(
    r"^(?:Here is (?:the|a) transcription[^\n]*\n|Here is what is written[^\n]*\n)",
    re.IGNORECASE,
)


def _strip_coordinates_and_noise(text: str) -> str:
    """Strips bounding box coordinates, grounding tags, and hallucinated quiz blocks."""
    if not text:
        return ""
    # Coordinate pairs: r'\(\s*\d+\s*,\s*\d+\s*\)\s*(?:,\s*\(\s*\d+\s*,\s*\d+\s*\))?'
    text = re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)\s*(?:,\s*\(\s*\d+\s*,\s*\d+\s*\))?", "", text)
    text = re.sub(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", "", text)
    # Grounding tags: r'<\|.*?\|>'
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<ref>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<box>.*?</box>", "", text, flags=re.DOTALL)
    # Hallucinated quiz blocks: r'(?i)(?:^|\n)\s*(?:#+\s*)?Question\s+\d+\s*:\s*What\b.*'
    text = re.sub(r"(?i)(?:^|\n)\s*(?:#+\s*)?Question\s+\d+\s*:\s*What\b.*", "", text)
    return text.strip()


def sanitize_ocr_text(text: str) -> str:
    """Post-process VLM output: drop tables, synthetic quizzes, sign-offs, loops, and coordinates."""
    if not text:
        return ""

    text = _strip_coordinates_and_noise(text)

    # Drop Markdown tables and separator rows
    text = re.sub(r"(?:^[ \t]*\|[^\n]+\|[ \t]*(?:\r?\n|$))+", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?[ \t]*$", "", text, flags=re.MULTILINE)

    # Cut off synthetic follow-up study quizzes
    text = QUIZ_CUTOFF_RE.split(text, maxsplit=1)[0]

    # Cut off conversational prefixes and sign-offs
    text = PREFIX_RE.sub("", text)
    text = SIGN_OFF_RE.split(text, maxsplit=1)[0]

    # Collapse repetitive word and line loops
    text = collapse_line_loops(text)
    text = collapse_word_loops(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


class LocalGradingEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self._lock = threading.Lock()

    def load_model(self):
        try:
            _ensure_torch()
        except Exception as exc:
            raise RuntimeError("PyTorch / Qwen2-VL is not installed. Install CUDA torch to run OCR.") from exc
        if self.model is None:
            print("[*] Loading Qwen2-VL-2B-Instruct into RTX 4050 VRAM...")
            try:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    local_files_only=True,
                )
                self.processor = AutoProcessor.from_pretrained(
                    MODEL_ID,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1024 * 28 * 28,
                    local_files_only=True,
                )
            except Exception:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self.processor = AutoProcessor.from_pretrained(
                    MODEL_ID,
                    min_pixels=256 * 28 * 28,
                    max_pixels=1024 * 28 * 28,
                )

    def unload_model(self):
        with self._lock:
            if self.model is not None:
                print("[*] Releasing GPU VRAM...")
                del self.model
                del self.processor
                self.model = None
                self.processor = None
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def telemetry(self) -> Dict:
        cuda = False
        name = "CPU"
        vram = 0.0
        try:
            _ensure_torch()
            cuda = bool(torch.cuda.is_available())
            if cuda:
                name = torch.cuda.get_device_name(0)
                vram = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        except Exception:
            pass
        return {
            "model": MODEL_ID,
            "device_name": name,
            "target_device": "NVIDIA RTX 4050 (6GB)",
            "cuda_available": cuda,
            "torch_available": TORCH_AVAILABLE,
            "model_loaded": self.model is not None,
            "vram_allocated_gb": vram,
        }

    def _generate_text(
        self,
        messages: List[Dict],
        max_tokens: int = 768,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        top_p: Optional[float] = None,
    ) -> str:
        with self._lock:
            self.load_model()
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Ensure sampling parameters are passed ONLY when do_sample=True
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
            }
            if temperature and temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                if top_p is not None:
                    gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            return sanitize_ocr_text(output.strip())

    def transcribe_images(self, image_paths: List[str], progress_cb=None) -> str:
        """Digitize handwritten booklet pages verbatim (question papers are teacher-entered)."""
        prompt_text = (
            "Read this handwritten exam page and write down all the handwritten English text line by line.\n"
            "- Keep margin question numbers like 'Q1] a]', 'b)', '2] (b)', '3] a', '3b)' and headings intact.\n"
            "- Do NOT output coordinates or numbers in parentheses like (25,47),(998,950).\n"
            "- Do NOT output Markdown tables or extra commentary."
        )
        pages_md = []
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                continue
            if progress_cb:
                progress_cb(f"Digitizing page {idx + 1}/{total} (verbatim)")
            print(f"[*] OCR Verbatim Processing Page {idx + 1}/{total}: {img_path}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            page_text = self._generate_text(
                messages, max_tokens=1024, temperature=0.0, repetition_penalty=1.05
            )

            # Fallback Retry: If page text after stripping coordinates is < 15 chars, retry once
            clean_check = _strip_coordinates_and_noise(page_text)
            if len(clean_check) < 15:
                print(f"[*] Page {idx + 1} produced low text or coordinates only (<15 chars). Running fallback retry...")
                retry_prompt = "Extract all handwritten words and sentences visible on this lined paper as plain text."
                retry_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": retry_prompt},
                        ],
                    }
                ]
                page_text = self._generate_text(
                    retry_messages, max_tokens=1024, temperature=0.0, repetition_penalty=1.05
                )

            pages_md.append(f"### Page {idx + 1}\n{page_text}")
        return "\n\n---\n\n".join(pages_md)

    def evaluate_answer(
        self,
        question_text: str,
        max_marks: float,
        key_points: List[Union[Dict, str]],
        student_answer: str,
        verified_examples: Optional[List[Dict]] = None,
    ) -> Dict:
        """Evaluates student answer against rubric points using EduEval deterministic grader."""
        from scorer import evaluate_with_edueval
        return evaluate_with_edueval(
            question_text=question_text,
            max_marks=max_marks,
            key_points=key_points,
            student_answer=student_answer,
            verified_examples=verified_examples,
        )
