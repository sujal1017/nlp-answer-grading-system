import threading
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "microsoft/trocr-base-handwritten"


def segment_handwriting_lines(image_path: str) -> List[Image.Image]:
    """Preprocess one page and return text-line crops in reading order."""
    page = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if page is None:
        raise ValueError(f"Unable to read image file: {image_path}")

    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if min(height, width) < 16:
        raise ValueError(f"Image is too small to OCR: {image_path}")

    block_size = min(31, min(height, width) // 2 * 2 - 1)
    block_size = max(3, block_size if block_size % 2 else block_size - 1)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        15,
    )

    rule_width = min(width, max(31, width // 14))
    rule_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (rule_width, 1))
    horizontal_rules = cv2.morphologyEx(binary, cv2.MORPH_OPEN, rule_kernel)
    rule_rows = np.count_nonzero(horizontal_rules, axis=1) > int(width * 0.25)
    rule_rows = cv2.dilate(
        rule_rows.astype(np.uint8),
        np.ones((5, 1), dtype=np.uint8),
    ).ravel().astype(bool)
    horizontal_rules[rule_rows, :] = 255
    detected_lines = cv2.HoughLinesP(
        binary,
        1,
        np.pi / 180,
        threshold=max(50, width // 8),
        minLineLength=max(100, width // 4),
        maxLineGap=max(10, width // 12),
    )
    line_positions = []
    if detected_lines is not None:
        for x1, y1, x2, y2 in detected_lines.reshape(-1, 4):
            line_width = abs(int(x2) - int(x1))
            line_height = abs(int(y2) - int(y1))
            if line_width >= width * 0.45 and line_height <= line_width * 0.05:
                line_positions.append(((int(y1) + int(y2)) // 2, line_width))
                cv2.line(
                    horizontal_rules,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    255,
                    2,
                )
    ink = cv2.subtract(binary, horizontal_rules)

    line_positions.sort()
    position_clusters = []
    for line_position, line_width in line_positions:
        if position_clusters and line_position - position_clusters[-1][-1][0] <= 18:
            position_clusters[-1].append((line_position, line_width))
        else:
            position_clusters.append([(line_position, line_width)])
    rule_centers = [
        int(np.average(
            [position for position, _ in cluster],
            weights=[line_width for _, line_width in cluster],
        ))
        for cluster in position_clusters
    ]
    rule_gaps = [
        current - previous
        for previous, current in zip(rule_centers, rule_centers[1:])
        if 35 <= current - previous <= 90
    ]
    use_rule_positions = len(rule_gaps) >= 3

    if use_rule_positions:
        line_pitch = float(np.median(rule_gaps))
        groups = []
        for index, center in enumerate(rule_centers):
            top = max(0, int(center - line_pitch * 0.65))
            bottom = min(height, int(center + line_pitch * 0.35))
            if index:
                top = max(top, (rule_centers[index - 1] + center) // 2)
            if index + 1 < len(rule_centers):
                bottom = min(bottom, (center + rule_centers[index + 1]) // 2)
            groups.append((top, bottom))
    else:
        row_ink = np.count_nonzero(ink, axis=1)
        active_rows = row_ink > max(5, int(width * 0.01))
        row_indices = np.flatnonzero(active_rows)
        if not len(row_indices):
            return []

        gap_limit = max(10, int(height * 0.008))
        groups = []
        start = previous = int(row_indices[0])
        for row in row_indices[1:]:
            row = int(row)
            if row - previous > gap_limit:
                groups.append((start, previous + 1))
                start = row
            previous = row
        groups.append((start, previous + 1))

    y_padding = max(6, int(height * 0.005))
    x_padding = max(8, int(width * 0.01))
    line_images = []
    for top, bottom in groups:
        if bottom - top < 5:
            continue
        line_ink = ink[top:bottom]
        if np.count_nonzero(line_ink) < max(80, int(width * 0.3)):
            continue
        columns = np.flatnonzero(np.count_nonzero(line_ink, axis=0))
        if not len(columns):
            continue

        left = max(0, int(columns[0]) - x_padding)
        right = min(width, int(columns[-1]) + 1 + x_padding)
        if not use_rule_positions:
            top = max(0, top - y_padding)
            bottom = min(height, bottom + y_padding)
        crop = gray[top:bottom, left:right]
        line_images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)))

    return line_images


class HandwritingOCREngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.torch = None
        self.device = "cpu"
        self._lock = threading.RLock()

    def load_model(self):
        if self.model is not None:
            return

        with self._lock:
            if self.model is not None:
                return

            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            processor = TrOCRProcessor.from_pretrained(MODEL_ID)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                try:
                    model.to(device)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    device = "cpu"
                    model.to(device)
            else:
                model.to(device)

            model.eval()
            self.processor = processor
            self.model = model
            self.torch = torch
            self.device = device

    def _transcribe_line(self, image: Image.Image) -> str:
        self.load_model()
        with self._lock:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            try:
                with self.torch.inference_mode():
                    generated = self.model.generate(pixel_values, max_new_tokens=128, num_beams=1)
            except RuntimeError as exc:
                if self.device != "cuda" or "out of memory" not in str(exc).lower():
                    raise
                self.model.to("cpu")
                self.device = "cpu"
                self.torch.cuda.empty_cache()
                pixel_values = pixel_values.cpu()
                with self.torch.inference_mode():
                    generated = self.model.generate(pixel_values, max_new_tokens=128, num_beams=1)
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def transcribe_images(
        self,
        image_paths: List[str],
        progress_cb: Optional[Callable[[str], None]] = None,
        auto_unload: bool = True,
    ) -> str:
        if not image_paths:
            raise ValueError("No pages were supplied for OCR.")

        pages = []
        total_pages = len(image_paths)
        try:
            for page_index, image_path in enumerate(image_paths, start=1):
                if not Path(image_path).is_file():
                    raise FileNotFoundError(f"OCR page does not exist: {image_path}")
                if progress_cb:
                    progress_cb(f"TrOCR: segmenting page {page_index}/{total_pages}")

                lines = segment_handwriting_lines(image_path)
                transcribed_lines = []
                for line_index, line_image in enumerate(lines, start=1):
                    if progress_cb:
                        progress_cb(
                            f"TrOCR: reading page {page_index}/{total_pages}, "
                            f"line {line_index}/{len(lines)}"
                        )
                    line_text = self._transcribe_line(line_image)
                    if line_text:
                        transcribed_lines.append(line_text)

                pages.append(f"### Page {page_index}\n" + "\n".join(transcribed_lines))
            return "\n\n".join(pages).strip()
        finally:
            if auto_unload:
                self.unload_model()

    def unload_model(self):
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.processor = None
                if self.torch is not None and self.device == "cuda":
                    self.torch.cuda.empty_cache()
            self.device = "cpu"
