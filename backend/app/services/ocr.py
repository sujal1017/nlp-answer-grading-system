"""OCR engines behind one interface.

Engines:
  tesseract - classical OCR. Good for PRINTED text (question papers). Poor on
              handwriting; pages OCR'd by it are always flagged for human review.
  qwen2vl   - Qwen2-VL vision-language model (optional, heavy; needs torch +
              transformers + model weights). Loaded lazily and cached; never
              required to start the application.

Confidence is reported only when the engine genuinely provides it (Tesseract word
confidences). Qwen2-VL provides none, so confidence is None - never invented.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol
import threading

import numpy as np


class OCRUnavailable(RuntimeError):
    """Engine cannot run in this environment (missing package, model, or binary)."""


@dataclass
class OCRResult:
    text: str
    engine: str
    confidence: Optional[float] = None  # 0-100 scale as given by the engine, or None
    warnings: list = field(default_factory=list)


class OCREngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OCRResult: ...


class TesseractEngine:
    name = "tesseract"
    LOW_CONFIDENCE = 60.0

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def _check(self):
        import shutil

        if shutil.which("tesseract") is None:
            raise OCRUnavailable("the 'tesseract' binary is not installed or not on PATH")

    def recognize(self, image: np.ndarray) -> OCRResult:
        self._check()
        import pytesseract

        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
        words, confs, lines = [], [], {}
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            words.append(word)
            if conf >= 0:
                confs.append(conf)
        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        mean_conf = round(float(np.mean(confs)), 1) if confs else None
        warnings = ["Tesseract is designed for printed text; handwriting needs careful human review."]
        if not words:
            warnings.append("No text recognised on this page.")
        elif mean_conf is not None and mean_conf < self.LOW_CONFIDENCE:
            warnings.append(f"Low OCR confidence ({mean_conf}); transcription is likely unreliable.")
        return OCRResult(text=text, engine=self.name, confidence=mean_conf, warnings=warnings)


class QwenVLEngine:
    """Qwen2-VL handwriting/printed OCR. Heavy: runs on GPU if available, else CPU (slow)."""

    name = "qwen2vl"
    PROMPT = (
        "Transcribe all text on this page exactly as written, preserving question numbers "
        "and labels such as 1a), Q2 (b), 3] a. Output plain text only. Write [illegible] for any "
        "word you cannot read - never guess. Do not output tables or bounding boxes."
    )

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError as exc:
            raise OCRUnavailable(
                "Qwen2-VL needs 'torch' and 'transformers' (see README, optional OCR extras)"
            ) from exc
        try:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
            )
        except Exception as exc:
            self._model = self._processor = None
            raise OCRUnavailable(f"could not load model '{self.model_id}': {exc}") from exc

    def unload(self):
        with self._lock:
            self._model = self._processor = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def recognize(self, image: np.ndarray) -> OCRResult:
        from PIL import Image

        with self._lock:
            self._load()
            import torch

            pil = Image.fromarray(image)
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}
            ]
            prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[prompt], images=[pil], return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.15
                )
            trimmed = out[:, inputs["input_ids"].shape[1]:]
            text = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        warnings = ["Vision-language OCR provides no confidence score; review the transcription."]
        if "[illegible]" in text.lower():
            warnings.append("Model marked some words as illegible.")
        return OCRResult(text=text, engine=f"{self.name}:{self.model_id}", confidence=None, warnings=warnings)


class RemoteVisionEngine:
    """OCR through any OpenAI-compatible *vision* chat endpoint (e.g. Qwen2-VL served by vLLM/Ollama,
    or a hosted VLM). Lets handwriting OCR run without a local GPU.

    PRIVACY: page images are sent to the configured server. Configure via environment:
    REMOTE_VLM_URL, REMOTE_VLM_MODEL, REMOTE_VLM_KEY (never stored in the database).
    """

    name = "remote_vlm"

    def __init__(self, url: str, model: str, api_key: str = "", timeout: int = 180):
        if not (url and model):
            raise OCRUnavailable("REMOTE_VLM_URL and REMOTE_VLM_MODEL must be set to use the remote_vlm engine")
        self.url, self.model, self.api_key, self.timeout = url, model, api_key, timeout

    def _post(self, payload: dict, headers: dict) -> dict:
        import requests

        r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def recognize(self, image: np.ndarray) -> OCRResult:
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {"model": self.model, "temperature": 0, "max_tokens": 1500, "messages": [{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": QwenVLEngine.PROMPT}]}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            text = self._post(payload, headers)["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise OCRUnavailable(f"remote vision service failed: {exc}") from exc
        warnings = ["Remote vision OCR provides no confidence score; review the transcription."]
        if "[illegible]" in text.lower():
            warnings.append("Model marked some words as illegible.")
        return OCRResult(text=text, engine=f"{self.name}:{self.model}", confidence=None, warnings=warnings)


_engines: dict = {}
_engines_lock = threading.Lock()


def get_engine(name: str, model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> OCREngine:
    """Return a cached engine instance (models are never reloaded per page)."""
    key = (name, model_id if name == "qwen2vl" else "")
    with _engines_lock:
        if key not in _engines:
            if name == "tesseract":
                _engines[key] = TesseractEngine()
            elif name == "qwen2vl":
                _engines[key] = QwenVLEngine(model_id)
            elif name == "remote_vlm":
                import os

                _engines[key] = RemoteVisionEngine(os.environ.get("REMOTE_VLM_URL", ""),
                                                   os.environ.get("REMOTE_VLM_MODEL", ""),
                                                   os.environ.get("REMOTE_VLM_KEY", ""))
            else:
                raise OCRUnavailable(f"unknown OCR engine '{name}'")
        return _engines[key]


def register_engine_for_tests(name: str, engine, model_id: str = ""):
    with _engines_lock:
        _engines[(name, model_id)] = engine
