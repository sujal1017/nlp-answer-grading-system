import os
import re
from typing import Dict, List
import fitz  # PyMuPDF

def convert_pdf_to_images(pdf_bytes: bytes, output_dir: str) -> List[str]:
    """Converts an uploaded PDF file into individual PNG images per page."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths

def clean_page_markers(markdown_text: str) -> str:
    cleaned = re.sub(r'###\s*Page\s*\d+\s*', '\n', markdown_text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n---\n', '\n', cleaned)
    return cleaned.strip()

def parse_sdmcet_question_paper(qp_markdown: str) -> List[Dict]:
    """
    Parses both SDMCET formats:
    Format A: 'Q1 (a) Define research... 5 Marks CO-1'
    Format B: 'Q. No. 1' on one line followed by 'a) ... [5M, CO:1]'
    """
    text = clean_page_markers(qp_markdown)
    extracted_questions = []

    # 1. Try Format A: Inline Q1 (a), Q1 (b), Q2 (a), etc.
    inline_pattern = re.compile(
        r'(?:^|\n)\s*(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\s*[\(\[\.\s]*([a-dA-D])\s*[\)\]\.\:\-]\s*(.+?)(?=(?:\n\s*(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*\d+\s*[\(\[\.\s]*[a-dA-D])|\n\s*OR\b|\Z)',
        re.IGNORECASE | re.DOTALL
    )
    marks_pattern = re.compile(r'[\[\(]?\s*(\d{1,2})\s*(?:Marks?|M)\b', re.IGNORECASE)
    co_pattern = re.compile(r'CO\s*[\:\-]?\s*(\d+)', re.IGNORECASE)

    for m in inline_pattern.finditer(text):
        q_num = m.group(1)
        sub_let = m.group(2).upper()
        body = m.group(3).strip()
        # Remove trailing 'OR' if attached
        body = re.sub(r'\bOR\s*$', '', body, flags=re.IGNORECASE).strip()

        max_marks = 5
        m_match = marks_pattern.search(body)
        if m_match:
            max_marks = int(m_match.group(1))
            body = body[:m_match.start()].strip()

        co_val = "CO:1"
        co_match = co_pattern.search(m.group(3))
        if co_match:
            co_val = f"CO:{co_match.group(1)}"

        body = re.sub(r'\s+', ' ', body).strip(' .-[]()')
        extracted_questions.append({
            "question_id": f"Q{q_num}{sub_let}",
            "question_text": body,
            "max_marks": max_marks,
            "co": co_val
        })

    if extracted_questions:
        return extracted_questions

    # 2. Fallback to Format B: Hierarchical 'Q. No. 1' -> 'a)'
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    main_q_pattern = re.compile(r'^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\b', re.IGNORECASE)
    sub_q_pattern = re.compile(r'^([a-dA-D])\s*[\)\.\]]\s*(.+)', re.IGNORECASE)

    current_main_q = "1"
    current_sub_q = None

    for line in lines:
        if any(skip in line.upper() for skip in ["USN", "COURSE CODE", "INTERNAL ASSESSMENT", "NOTE:", "*****"]):
            continue
        main_match = main_q_pattern.match(line)
        if main_match:
            if current_sub_q:
                extracted_questions.append(current_sub_q)
                current_sub_q = None
            current_main_q = main_match.group(1)
            continue

        sub_match = sub_q_pattern.match(line)
        if sub_match:
            if current_sub_q:
                extracted_questions.append(current_sub_q)
            sub_letter = sub_match.group(1).upper()
            q_text = sub_match.group(2).strip()
            current_sub_q = {
                "question_id": f"Q{current_main_q}{sub_letter}",
                "question_text": q_text,
                "max_marks": 5,
                "co": "CO:1"
            }
        elif current_sub_q:
            current_sub_q["question_text"] += " " + line

        if current_sub_q:
            m_match = marks_pattern.search(current_sub_q["question_text"])
            if m_match:
                current_sub_q["max_marks"] = int(m_match.group(1))
                current_sub_q["question_text"] = current_sub_q["question_text"][:m_match.start()].strip()

    if current_sub_q:
        extracted_questions.append(current_sub_q)

    return extracted_questions

def _match_orphan_subquestion(answer_text: str, sub_letter: str, rubrics: List[Dict], existing_keys: List[str]) -> str:
    """
    If a student writes 'a) The different types of research' without the main number '2',
    this compares the first 250 chars of the answer with all rubric questions ending in 'A' (Q1A, Q2A, Q3A)
    and picks the rubric with the highest keyword overlap.
    """
    candidates = [r for r in rubrics if r["question_id"].endswith(sub_letter)]
    if not candidates:
        return f"Q1{sub_letter}"

    ans_words = set(re.findall(r'[a-z]{4,}', answer_text[:300].lower()))
    best_qid = candidates[0]["question_id"]
    best_overlap = -1

    for c in candidates:
        q_words = set(re.findall(r'[a-z]{4,}', c["question_text"].lower()))
        overlap = len(ans_words.intersection(q_words))
        # Prefer questions not yet assigned if overlap is tied
        if overlap > best_overlap or (overlap == best_overlap and c["question_id"] not in existing_keys):
            best_overlap = overlap
            best_qid = c["question_id"]

    return best_qid

def chunk_student_booklet(raw_markdown: str, rubrics: List[Dict] = None) -> Dict[str, str]:
    """
    Splits a student booklet transcription into Q1A, Q1B, Q2A, Q2B, Q3A, Q3B.
    Handles handwritten OCR quirks:
      - '2] (b)' or '2] (6)' -> Q2B
      - '3] a' or '3) a'     -> Q3A
      - '3b)' or '36)'       -> Q3B
      - orphan 'a)' at start -> matched against rubric keywords (e.g., Q2A)
    """
    text = clean_page_markers(raw_markdown)
    # Strip top 'Internal Assessment - I' banner if present
    text = re.sub(r'^\s*\|?\s*Internal Assessment\s*[-–—]?\s*[I1V]+\s*\n?', '', text, flags=re.IGNORECASE)

    # Normalize common OCR handwriting confusions for question headers at start of lines:
    # e.g., '2] (6)' -> '2(b)', '36)' -> '3b)', '3] a' -> '3a)'
    text = re.sub(r'(?:^|\n)\s*([1-3])\s*[\]\)\.]?\s*\(\s*6\s*\)', r'\nQ\1b) ', text)
    text = re.sub(r'(?:^|\n)\s*([1-3])6\s*[\)\]]', r'\nQ\1b) ', text)
    text = re.sub(r'(?:^|\n)\s*([1-3])\s*[\]\)]\s*([a-dA-D])\b', r'\nQ\1\2) ', text)

    # Match explicit question headers: Q1a, 2(b), 3a), OR standalone 'a)' at the very start of the booklet
    header_pattern = re.compile(
        r'(?:^|\n)\s*(?:#+\s*)?(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?(?:([1-3])\s*[\]\)\.\-\s]*[\(\[]?\s*([a-dA-D])\s*[\)\]\.\:\-]|\A\s*([a-bA-B])\s*[\)\]\.])\s*',
        re.IGNORECASE
    )

    matches = list(header_pattern.finditer(text))
    if not matches:
        default_qid = rubrics[0]["question_id"].upper() if rubrics else "Q1A"
        return {default_qid: text}

    chunks: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        q_num = match.group(1)
        sub_let = (match.group(2) or match.group(3) or "A").upper()

        start_pos = match.end()
        end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start_pos:end_pos].strip()

        if q_num:
            q_id = f"Q{q_num}{sub_let}"
        else:
            # Orphan sub-question like 'a)' at the start of Page 1 -> match using rubric keywords
            q_id = _match_orphan_subquestion(content, sub_let, rubrics or [], list(chunks.keys()))

        if q_id in chunks:
            chunks[q_id] += f"\n\n{content}"
        else:
            chunks[q_id] = content

    return chunks

def calculate_ia_total(scores_dict: Dict[str, float]) -> float:
    """
    SDMCET IA Rule: Question 3 is Compulsory; Answer any One Full question from Q1 and Q2.
    """
    q1_total = scores_dict.get("Q1A", 0.0) + scores_dict.get("Q1B", 0.0)
    q2_total = scores_dict.get("Q2A", 0.0) + scores_dict.get("Q2B", 0.0)
    q3_total = scores_dict.get("Q3A", 0.0) + scores_dict.get("Q3B", 0.0)

    best_choice = max(q1_total, q2_total)
    return round(best_choice + q3_total, 2)