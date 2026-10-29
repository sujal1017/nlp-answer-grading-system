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
        """Loads Qwen2-VL-2B into RTX 4050 VRAM with balanced pixel limits."""
        if self.model is None:
            print("[*] Loading Qwen2-VL-2B-Instruct into RTX 4050 VRAM...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            # Limit max_pixels so high-res PDF pages don't exhaust 6GB VRAM or trigger box tokens
            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28
            )

    def unload_model(self):
        if self.model is not None:
            print("[*] Releasing GPU VRAM...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

    def _generate_text(
        self,
        messages: List[Dict],
        max_tokens: int = 768,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15
    ) -> str:
        """Runs inference with repetition_penalty to prevent infinite table loops."""
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

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty
        }
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
        """Runs OCR across Question Paper or Student Booklet pages without bounding boxes or tables."""
        pages_md = []
        if is_question_paper:
            prompt_text = (
                "Read this exam question paper image and transcribe every question verbatim as plain text. "
                "Format each question on a new line starting with its number, such as:\n"
                "Q1 (a) <question text> 5 Marks CO-1\n"
                "Q1 (b) <question text> 5 Marks CO-2\n"
                "Do NOT output coordinates, bounding boxes, or Markdown tables."
            )
        else:
            prompt_text = (
                "Read this handwritten student answer page and transcribe the handwritten text accurately into plain paragraphs and bullet points. "
                "Preserve question numbers written in the margin such as 'a)', '2] (b)', '3] a', '3b)'. "
                "Do NOT use Markdown tables (`| --- |`) and do NOT repeat phrases."
            )

        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
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
            page_text = self._generate_text(messages, max_tokens=768, temperature=0.0, repetition_penalty=1.18)
            pages_md.append(f"### Page {idx + 1}\n{page_text}")

        return "\n\n---\n\n".join(pages_md)

    def generate_rubric_points(self, question_text: str, max_marks: int, num_points: int = 4) -> List[Dict]:
        """Generates 4 weighted rubric criteria for an extracted question."""
        print(f"[*] Generating initial rubric key points for: {question_text[:50]}...")
        prompt = f"""You are a university professor creating an evaluation scheme for "Research Methodology & IPR".
Question: "{question_text}"
Maximum Marks: {max_marks}

List exactly {num_points} core concepts or points a student should cover to earn full marks.
Respond ONLY with a valid JSON array of {num_points} strings:
["Concept 1", "Concept 2", "Concept 3", "Concept 4"]"""

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate_text(messages, max_tokens=256, temperature=0.0, repetition_penalty=1.05)

        points_list = []
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    points_list = [str(p).strip() for p in parsed if str(p).strip()][:num_points]
            except json.JSONDecodeError:
                pass

        if not points_list:
            points_list = [
                f"Clear definition and core meaning of {question_text[:35]}",
                "Major classifications, steps, or essential features explained",
                "Explanation of sub-types or key variables with examples",
                "Logical structure and coverage of required concepts"
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
        Grades a student's answer chunk fairly (ignoring minor OCR spelling artifacts in handwritten scripts).
        """
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

        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
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

        # Take the higher of the calibrated rubric sum or LLM holistic score if the student wrote a lengthy answer
        final_score = max(weighted_score_sum, llm_score) if matched_indices else llm_score
        final_score = round(min(max(final_score, 0.0), float(max_marks)) * 2) / 2  # Round to nearest 0.5

        return {
            "matched_points": matched_points,
            "missing_points": missing_points,
            "ai_score": final_score,
            "ai_feedback": feedback
        }