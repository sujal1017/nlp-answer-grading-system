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


def sanitize_ocr_text(text: str) -> str:
    """Post-process VLM output: drop tables, synthetic quizzes, sign-offs, loops, and visual grounding coordinates."""
    if not text:
        return ""

    # Drop visual grounding bounding box coordinates like (25,47),(998,950) or [25, 47, 998, 950]
    text = re.sub(r"\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\)\s*,\s*\(\s*\d{1,4}\s*,\s*\d{1,4}\s*\)", "", text)
    text = re.sub(r"\[\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*\]", "", text)
    text = re.sub(r"<ref>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<box>.*?</box>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|box_start\|>.*?<\|box_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|object_ref_start\|>.*?<\|object_ref_end\|>", "", text, flags=re.DOTALL)

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
        """Digitize handwritten booklet pages verbatim (question papers are teacher-entered).

        Uses a two-message structure (system + user roles) to enforce strict
        verbatim OCR without visual grounding hallucinations or markdown tables.
        """
        system_prompt = (
            "You are a strict verbatim Optical Character Recognition (OCR) engine. "
            "You transcribe handwritten text exactly as written, without any additions, "
            "corrections, summaries, or editorial changes."
        )
        user_prompt = (
            "Transcribe the handwritten English text on this exam paper line by line.\n"
            "CRITICAL RULES:\n"
            "- Output ONLY the transcribed words.\n"
            "- NEVER output bounding box coordinates, pixel coordinates, numbers in parentheses like (25,47),(998,950), or detection tags.\n"
            "- Do NOT format as a Markdown table.\n"
            "- If handwriting starts with a margin number like 'Q1] a]', include it at the start."
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
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            page_text = self._generate_text(
                messages, max_tokens=1024, temperature=0.0, repetition_penalty=1.05
            )
            pages_md.append(f"### Page {idx + 1}\n{page_text}")
        return "\n\n---\n\n".join(pages_md)

    def evaluate_answer(
        self,
        question_text: str,
        max_marks: int,
        key_points: List[Union[Dict, str]],
        student_answer: str,
        verified_examples: Optional[List[Dict]] = None,
    ) -> Dict:
        norm_points = []
        default_w = round(float(max_marks) / max(len(key_points), 1), 2)
        for item in key_points:
            if isinstance(item, dict):
                norm_points.append({"point": item["point"], "weight": float(item.get("weight", default_w))})
            else:
                norm_points.append({"point": str(item), "weight": default_w})

        if not student_answer or len(student_answer.strip()) < 10:
            return {
                "matched_points": [],
                "missing_points": [p["point"] for p in norm_points],
                "ai_score": 0.0,
                "ai_feedback": "No answer written or transcription is empty.",
            }

        rubric_numbered = "\n".join(
            [f"{i + 1}. {p['point']} (Weight: {p['weight']} marks)" for i, p in enumerate(norm_points)]
        )

        few_shot_block = ""
        if verified_examples:
            examples_str = []
            for ex in verified_examples:
                examples_str.append(
                    f"- Past Student Answer: \"{ex['transcribed_text'][:200]}...\"\n"
                    f"  Teacher Approved Score: {ex['teacher_score']} / {max_marks}\n"
                    f"  Teacher Feedback: {ex['ai_feedback']}"
                )
            few_shot_block = "\nPast Teacher Grading Reference:\n" + "\n".join(examples_str) + "\n"

        prompt = f"""You are a supportive university professor grading a handwritten Internal Assessment answer sheet.
Ignore minor spelling or OCR transcription errors if the student's conceptual understanding is clear.

Question: {question_text}
Maximum Marks: {max_marks}

Rubric Key Points:
{rubric_numbered}
{few_shot_block}
Student's Handwritten Answer (OCR Transcribed):
\"\"\"{student_answer}\"\"\"

Instructions:
1. Identify which rubric points (1 to {len(norm_points)}) the student has addressed (even if phrased in their own words).
2. Award full or near-full marks if the student has written a detailed, structured answer covering multiple sub-points.
3. Do NOT output Markdown tables. Do NOT invent extra quiz questions.
Respond ONLY with a valid JSON object in this exact format:
{{
  "matched_indices": [1, 2, 3, 4],
  "llm_suggested_score": 4.5,
  "feedback": "Concise 2-sentence summary of strengths and any minor omissions."
}}"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate_text(messages, max_tokens=300, temperature=0.0, repetition_penalty=1.05)

        matched_indices = []
        llm_score = 0.0
        feedback = "Evaluated against question rubric."

        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                matched_indices = [int(i) for i in data.get("matched_indices", []) if str(i).isdigit()]
                llm_score = float(data.get("llm_suggested_score", 0.0))
                feedback = str(data.get("feedback", feedback))
            except Exception:
                feedback = f"Evaluated answer. Raw notes: {raw_output[:150]}"

        matched_points = []
        missing_points = []
        weighted_score_sum = 0.0
        for idx, item in enumerate(norm_points):
            pt_num = idx + 1
            if pt_num in matched_indices:
                matched_points.append(item["point"])
                weighted_score_sum += item["weight"]
            else:
                missing_points.append(item["point"])

        final_score = max(weighted_score_sum, llm_score) if matched_indices else llm_score
        final_score = round(min(max(final_score, 0.0), float(max_marks)) * 2) / 2

        return {
            "matched_points": matched_points,
            "missing_points": missing_points,
            "ai_score": final_score,
            "ai_feedback": feedback,
        }
