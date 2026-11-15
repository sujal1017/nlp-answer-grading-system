import json
import os
import re
import threading
from typing import Dict, List, Optional, Tuple, Union

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
    """Strips bounding box coordinates, grounding tags, and synthetic quiz blocks without deleting student text."""
    if not text:
        return ""
    # Coordinate pairs: r'\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\)\s*(?:,\s*\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\))?'
    text = re.sub(
        r"\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\)\s*(?:,\s*\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\))?",
        "",
        text,
    )
    # Coordinate brackets: r'\[\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*\]'
    text = re.sub(r"\[\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*\]", "", text)
    # Grounding tags: r'<\|.*?\|>' and non-DOTALL reference/box tags
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"</?(?:ref|box|quad)>", "", text)
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


def _preprocess_page_image(image_path: str) -> str:
    """Enhance contrast and sharpness for faint handwritten blue ink and lined backgrounds."""
    if not os.path.exists(image_path):
        return image_path
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import tempfile
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            contrasted = ImageOps.autocontrast(rgb_img, cutoff=1)
            contrasted = ImageEnhance.Contrast(contrasted).enhance(1.35)
            sharpened = ImageEnhance.Sharpness(contrasted).enhance(1.5)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            sharpened.save(tmp_path)
            return tmp_path
    except Exception as exc:
        print(f"[!] Warning: image preprocessing failed for {image_path}: {exc}")
        return image_path


def _crop_horizontal_halves(image_path: str) -> Tuple[str, str]:
    """Crops the image horizontally into two overlapping halves (0-55% and 45-100% height)."""
    from PIL import Image
    import tempfile
    with Image.open(image_path) as img:
        w, h = img.size
        top_box = (0, 0, w, int(h * 0.55))
        bottom_box = (0, int(h * 0.45), w, h)
        top_half = img.crop(top_box)
        bottom_half = img.crop(bottom_box)

        t1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        p1 = t1.name
        t1.close()
        top_half.save(p1)

        t2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        p2 = t2.name
        t2.close()
        bottom_half.save(p2)

        return p1, p2


def extract_left_margin_crop(image_path: str, margin_ratio: float = 0.18) -> Optional[str]:
    """Crops the left margin_ratio (default 18%) of an image using Pillow and returns the temporary cropped image path."""
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image
        import tempfile
        with Image.open(image_path) as img:
            w, h = img.size
            crop_w = max(10, int(w * margin_ratio))
            margin_crop = img.crop((0, 0, crop_w, h))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            margin_crop.save(tmp_path)
            return tmp_path
    except Exception as exc:
        print(f"[!] Warning: margin crop failed: {exc}")
        return None


def extract_left_margin_marker(
    image_path: str,
    margin_ratio: float = 0.18,
    engine: Optional["LocalGradingEngine"] = None,
) -> Optional[str]:
    """Crops the left 18% vertical margin using Pillow and extracts margin question markers."""
    crop_path = extract_left_margin_crop(image_path, margin_ratio)
    if not crop_path:
        return None
    try:
        if engine is not None:
            prompt = (
                "Look at this left margin strip of an exam booklet. "
                "Output ONLY the question number marker written here (e.g. 'Q1] a', '2] (b)', '3] a', '3b)'). "
                "If no question marker is present, output None."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": crop_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            res = engine._generate_text(messages, max_tokens=32, temperature=0.0)
            res = _strip_coordinates_and_noise(res).strip()
            if res and res.lower() != "none" and len(res) <= 30:
                return res
        return None
    finally:
        if os.path.exists(crop_path):
            os.remove(crop_path)


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

            # Grounding block via bad_words_ids
            grounding_tokens = ["<|box_start|>", "<|box_end|>", "<|object_ref_start|>", "<|object_ref_end|>"]
            bad_words_ids = []
            if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
                for t in grounding_tokens:
                    try:
                        tid = self.processor.tokenizer.convert_tokens_to_ids(t)
                        if tid is not None and tid != getattr(self.processor.tokenizer, "unk_token_id", None):
                            bad_words_ids.append([tid])
                    except Exception:
                        pass

            # Ensure sampling parameters are passed ONLY when do_sample=True
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
            }
            if bad_words_ids:
                gen_kwargs["bad_words_ids"] = bad_words_ids

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

    def extract_left_margin_marker(self, image_path: str, margin_ratio: float = 0.18) -> Optional[str]:
        return extract_left_margin_marker(image_path, margin_ratio=margin_ratio, engine=self)

    def transcribe_images(
        self,
        image_paths: List[str],
        progress_cb=None,
        extract_margins: bool = False,
        auto_unload: bool = True,
    ) -> str:
        """Digitize handwritten booklet pages verbatim with 3-stage fallback guarantee (no page skipped)."""
        prompt_attempt1 = (
            "Transcribe all handwritten text from this exam answer sheet page from top to bottom. "
            "Include all headings, margin numbers (such as a), 2(b), 3a, 3b), bullet points, and definitions."
        )
        prompt_split = "Read and transcribe every handwritten line visible in this image:"
        pages_md = []
        total = len(image_paths)
        try:
            for idx, img_path in enumerate(image_paths):
                if not os.path.exists(img_path):
                    continue
                if progress_cb:
                    progress_cb(f"Digitizing page {idx + 1}/{total} (verbatim)")
                print(f"[*] OCR Verbatim Processing Page {idx + 1}/{total}: {img_path}")

                margin_marker = None
                if extract_margins:
                    margin_marker = self.extract_left_margin_marker(img_path)

                # Preprocess image contrast & sharpness for faint blue ink
                enhanced_path = _preprocess_page_image(img_path)
                cleanup_files = []
                if enhanced_path != img_path:
                    cleanup_files.append(enhanced_path)

                page_text = ""
                words_count = 0
                try:
                    # Attempt 1: Full-page OCR with clean positive prompt
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": enhanced_path},
                                {"type": "text", "text": prompt_attempt1},
                            ],
                        }
                    ]
                    page_text = self._generate_text(
                        messages, max_tokens=1024, temperature=0.0, repetition_penalty=1.05
                    )
                    words_count = len(re.findall(r'[a-zA-Z]{3,}', page_text))

                    # Attempt 2: Top/Bottom Split Pass if < 15 alphabetic words
                    if words_count < 15:
                        print(f"[*] Page {idx + 1} produced only {words_count} words (<15 words). Running Attempt 2 (Top/Bottom Split Pass)...")
                        try:
                            p_top, p_bot = _crop_horizontal_halves(enhanced_path)
                            cleanup_files.extend([p_top, p_bot])
                            top_text = self._generate_text(
                                [{"role": "user", "content": [{"type": "image", "image": p_top}, {"type": "text", "text": prompt_split}]}],
                                max_tokens=768, temperature=0.0, repetition_penalty=1.05
                            )
                            bot_text = self._generate_text(
                                [{"role": "user", "content": [{"type": "image", "image": p_bot}, {"type": "text", "text": prompt_split}]}],
                                max_tokens=768, temperature=0.0, repetition_penalty=1.05
                            )
                            candidate_text = f"{top_text}\n{bot_text}".strip()
                            candidate_words = len(re.findall(r'[a-zA-Z]{3,}', candidate_text))
                            if candidate_words > words_count:
                                page_text = candidate_text
                                words_count = candidate_words
                        except Exception as split_exc:
                            print(f"[!] Attempt 2 failed: {split_exc}")

                    # Attempt 3: Mild Sampling Fallback if still < 15 alphabetic words
                    if words_count < 15:
                        print(f"[*] Page {idx + 1} produced only {words_count} words (<15 words). Running Attempt 3 (Mild Sampling Fallback)...")
                        try:
                            sample_text = self._generate_text(
                                messages, max_tokens=1024, temperature=0.2, repetition_penalty=1.08
                            )
                            sample_words = len(re.findall(r'[a-zA-Z]{3,}', sample_text))
                            if sample_words > words_count:
                                page_text = sample_text
                                words_count = sample_words
                        except Exception as sample_exc:
                            print(f"[!] Attempt 3 failed: {sample_exc}")

                finally:
                    # Clean up temporary preprocessed/cropped files
                    for cf in cleanup_files:
                        if os.path.exists(cf):
                            try:
                                os.remove(cf)
                            except Exception:
                                pass

                # Log word count for every single page
                print(f"[*] Page {idx + 1}: {words_count} words")

                if margin_marker and margin_marker not in page_text:
                    page_text = f"{margin_marker}\n{page_text}"

                pages_md.append(f"--- Page {idx + 1} ---\n{page_text}")
            return "\n\n".join(pages_md)
        finally:
            if auto_unload:
                self.unload_model()

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
