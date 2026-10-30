import os
import re
import json
import torch
from typing import List, Dict, Union
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

class LocalGradingEngine:
    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self):
        """Loads Qwen2-VL-2B into GPU VRAM in bfloat16 only if not already loaded."""
        if self.model is None:
            print("[*] Loading Qwen2-VL-2B-Instruct into RTX 4050 VRAM...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    def unload_model(self):
        """Frees GPU VRAM after batch processing."""
        if self.model is not None:
            print("[*] Releasing GPU VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

    def _generate_text(self, messages: List[Dict], max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Internal helper to run multimodal or text-only inference."""
        self.load_model()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": max_tokens}
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output.strip()

    def transcribe_images(self, image_paths: List[str], is_question_paper: bool = False) -> str:
        """Runs OCR across one or more images (Question Paper or Student Booklet)."""
        pages_md = []
        prompt_text = (
            "Transcribe this university question paper verbatim into plain text. "
            "Keep question numbers like 'Q. No. 1', sub-questions 'a)', 'b)', and mark tags like '[5M, CO:1]' intact."
            if is_question_paper else
            "Transcribe all handwritten student answers in this image verbatim into clean Markdown. "
            "Preserve question numbers (e.g., Q1a, 1b, Ans 1), headers, and bullet points accurately."
        )

        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                print(f"[-] Warning: File not found -> {img_path}")
                continue
            print(f"[*] OCR Processing Page {idx + 1}/{len(image_paths)}: {img_path}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            page_text = self._generate_text(messages, max_tokens=1024, temperature=0.0)
            pages_md.append(f"### Page {idx + 1}\n{page_text}")

        return "\n\n---\n\n".join(pages_md)

    def generate_rubric_points(self, question_text: str, max_marks: int, num_points: int = 4) -> List[Dict]:
        """
        Auto-generates weighted evaluation criteria from an extracted Question Paper prompt.
        Each point receives an initial weight = max_marks / num_points.
        """
        print(f"[*] Generating initial rubric key points for: {question_text[:50]}...")
        prompt = f"""You are an engineering professor creating an evaluation scheme for an exam question.
Question: "{question_text}"
Maximum Marks: {max_marks}

Break the ideal answer down into exactly {num_points} distinct, atomic key points that a student must write to get full marks.
Respond ONLY with a valid JSON array of strings, like this:
["Point 1 description", "Point 2 description", "Point 3 description", "Point 4 description"]"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate_text(messages, max_tokens=256, temperature=0.1)

        points_list = []
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    points_list = [str(p).strip() for p in parsed if str(p).strip()]
            except json.JSONDecodeError:
                pass

        # Fallback if JSON parsing fails
        if not points_list:
            points_list = [
                f"Core definition and concept explanation for: {question_text[:40]}",
                "Detailed technical working, architecture, or classification",
                "Relevant diagram, illustration, or practical example",
                "Accurate conclusion or key properties"
            ]

        default_weight = round(float(max_marks) / len(points_list), 2)
        return [{"point": pt, "weight": default_weight} for pt in points_list]

    def evaluate_answer(
        self,
        question_text: str,
        max_marks: int,
        key_points: List[Union[Dict, str]],
        student_answer: str,
        verified_examples: List[Dict] = None
    ) -> Dict:
        """
        Evaluates a student's chunked answer using:
        1. Atomic rubric matching + live teacher-calibrated weights.
        2. Few-shot calibration from previous teacher-verified answers.
        """
        # Normalize key_points into weighted dicts
        norm_points = []
        default_w = round(float(max_marks) / max(len(key_points), 1), 2)
        for item in key_points:
            if isinstance(item, dict):
                norm_points.append({"point": item["point"], "weight": float(item.get("weight", default_w))})
            else:
                norm_points.append({"point": str(item), "weight": default_w})

        rubric_numbered = "\n".join(
            [f"{i+1}. {p['point']} (Weight: {p['weight']} marks)" for i, p in enumerate(norm_points)]
        )

        # Build Few-Shot Calibration block from past teacher approvals
        few_shot_block = ""
        if verified_examples:
            examples_str = []
            for ex in verified_examples:
                examples_str.append(
                    f"- Past Student Answer: \"{ex['transcribed_text'][:200]}...\"\n"
                    f"  Teacher Approved Score: {ex['teacher_score']} / {max_marks}\n"
                    f"  Teacher Feedback: {ex['ai_feedback']}"
                )
            few_shot_block = (
                "\nReference Past Teacher Grading for Calibration:\n"
                + "\n".join(examples_str) + "\n"
            )

        prompt = f"""You are a strict university examiner grading a student answer booklet.

Question: {question_text}
Maximum Marks: {max_marks}

Rubric Checklist (with calibrated weights):
{rubric_numbered}
{few_shot_block}
Student's Answer:
\"\"\"{student_answer}\"\"\"

Task:
Check which numbered rubric points (1 to {len(norm_points)}) are clearly answered by the student.
Respond ONLY with a valid JSON object in this exact format:
{{
  "matched_indices": [1, 2],
  "llm_suggested_score": 2.5,
  "feedback": "Concise 2-sentence explanation of what the student covered and what was missed."
}}"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate_text(messages, max_tokens=300, temperature=0.0)

        matched_indices = []
        llm_score = 0.0
        feedback = "Evaluated against rubric checklist."

        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                matched_indices = data.get("matched_indices", [])
                llm_score = float(data.get("llm_suggested_score", 0.0))
                feedback = str(data.get("feedback", feedback))
            except Exception:
                feedback = f"Partial parse. Model output: {raw_output[:150]}"

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

        # Combine calibrated rubric weight sum with LLM score if indices were matched
        final_score = weighted_score_sum if matched_indices else llm_score
        # Clamp strictly between 0 and max_marks (rounded to nearest 0.5 or 0.25)
        final_score = round(min(max(final_score, 0.0), float(max_marks)) * 4) / 4

        return {
            "matched_points": matched_points,
            "missing_points": missing_points,
            "ai_score": final_score,
            "ai_feedback": feedback
        }

if __name__ == "__main__":
    # Quick sanity test without loading the 4.5GB model until needed
    engine = LocalGradingEngine()
    print("[+] LocalGradingEngine initialized and ready.")