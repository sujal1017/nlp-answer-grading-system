import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import pymupdf as fitz

STOPWORDS = {
    "about", "after", "also", "and", "any", "are", "been", "briefly", "define",
    "describe", "discuss", "each", "explain", "following", "for", "from", "give",
    "have", "how", "into", "its", "list", "of", "or", "state", "the", "their",
    "them", "then", "this", "that", "with", "write", "what", "when", "which",
    "your", "using", "used", "than", "them", "they", "will", "were",
}


def convert_pdf_to_images(pdf_bytes: bytes, output_dir: str) -> List[str]:
    """Converts an uploaded PDF file into individual PNG images per page."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_path = str(Path(output_dir) / f"page_{page_num + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths


def clean_page_markers(markdown_text: str) -> str:
    cleaned = re.sub(r"###\s*Page\s*\d+\s*", "\n", markdown_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n---+\n", "\n", cleaned)
    return cleaned.strip()


def parse_sdmcet_question_paper(qp_markdown: str) -> List[Dict]:
    """Parses teacher-pasted SDMCET question text into Q1A/Q1B/... records."""
    text = clean_page_markers(qp_markdown)
    marks_pattern = re.compile(r"[\[(]?\s*(\d{1,2})\s*(?:Marks?|M)\b", re.IGNORECASE)
    co_pattern = re.compile(r"CO\s*[:\-]?\s*(\d+)", re.IGNORECASE)
    inline_line = re.compile(
        r"^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\s*[(\[]\s*([a-dA-D])\s*[)\]]\s*(.+)$",
        re.IGNORECASE,
    )
    inline_line_b = re.compile(
        r"^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\s+([a-dA-D])\s*[).:\-]\s*(.+)$",
        re.IGNORECASE,
    )

    extracted_questions: List[Dict] = []
    current = None

    def flush():
        nonlocal current
        if not current:
            return
        body = re.sub(r"\s+", " ", current["question_text"]).strip(" .-[]()")
        body = re.sub(r"\bOR\s*$", "", body, flags=re.IGNORECASE).strip()
        m_match = marks_pattern.search(body)
        if m_match:
            current["max_marks"] = int(m_match.group(1))
            body = body[: m_match.start()].strip()
        co_match = co_pattern.search(current["question_text"])
        if co_match:
            current["co"] = f"CO:{co_match.group(1)}"
        current["question_text"] = body
        extracted_questions.append(current)
        current = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("OR"):
            continue
        if any(skip in line.upper() for skip in ["USN", "COURSE CODE", "INTERNAL ASSESSMENT", "NOTE:", "*****"]):
            continue

        m = inline_line.match(line) or inline_line_b.match(line)
        if m:
            flush()
            current = {
                "question_id": f"Q{m.group(1)}{m.group(2).upper()}",
                "question_text": m.group(3).strip(),
                "max_marks": 5,
                "co": "CO:1",
            }
            continue

        main_match = re.match(r"^(?:#+\s*)?Q(?:uestion)?\.?\s*(?:No\.?)?\s*(\d+)\b", line, re.I)
        if main_match and not re.search(r"[a-dA-D]", line[main_match.end() : main_match.end() + 6] or ""):
            flush()
            current = {
                "question_id": f"Q{main_match.group(1)}A",
                "question_text": "",
                "max_marks": 5,
                "co": "CO:1",
                "_main": main_match.group(1),
                "_open": True,
            }
            rest = line[main_match.end() :].strip(" .:-")
            if rest:
                current["question_text"] = rest
            continue

        sub_match = re.match(r"^([a-dA-D])\s*[).\]\-]\s*(.+)$", line, re.I)
        if sub_match:
            main_n = "1"
            if current:
                prev_id = current["question_id"]
                main_n = re.sub(r"^Q(\d).*", r"\1", prev_id)
                flush()
            current = {
                "question_id": f"Q{main_n}{sub_match.group(1).upper()}",
                "question_text": sub_match.group(2).strip(),
                "max_marks": 5,
                "co": "CO:1",
            }
            continue

        if current:
            current["question_text"] = (current["question_text"] + " " + line).strip()

    flush()
    return extracted_questions


def parse_teacher_questions(raw_text: str) -> List[Dict]:
    """Accepts pasted exam text or compact `Q1A | prompt | 5` rows."""
    compact = re.compile(
        r"^\s*(Q?\s*[1-3]\s*[A-Da-d])\s*[|:.\-]\s*(.+?)\s*[|:]\s*(\d+(?:\.\d+)?)\s*$"
    )
    rows = []
    for line in raw_text.splitlines():
        m = compact.match(line.strip())
        if not m:
            continue
        qid = re.sub(r"\s+", "", m.group(1).upper())
        if not qid.startswith("Q"):
            qid = "Q" + qid
        rows.append(
            {
                "question_id": qid,
                "question_text": m.group(2).strip(),
                "max_marks": int(float(m.group(3))),
                "co": "CO:1",
            }
        )
    if rows:
        return rows
    return parse_sdmcet_question_paper(raw_text)


def keywords(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in STOPWORDS}


def _match_orphan_subquestion(answer_text: str, sub_letter: str, rubrics: List[Dict], existing_keys: List[str]) -> str:
    candidates = [r for r in rubrics if str(r.get("question_id", "")).upper().endswith(sub_letter.upper())]
    if not candidates:
        return f"Q1{sub_letter.upper()}"

    ans_words = keywords(answer_text[:400])
    best_qid = str(candidates[0]["question_id"]).upper()
    best_overlap = -1

    for c in candidates:
        q_words = keywords(c.get("question_text", ""))
        overlap = len(ans_words.intersection(q_words))
        qid = str(c["question_id"]).upper()
        if overlap > best_overlap or (overlap == best_overlap and qid not in existing_keys):
            best_overlap = overlap
            best_qid = qid
    return best_qid


def _semantic_heading_qid(line: str, rubrics: List[Dict]) -> Optional[str]:
    """Match a standalone student heading against teacher-configured question keywords."""
    stripped = re.sub(r"^[#>*\-\s]+", "", line).strip(" .:;-*_")
    word_count = len(stripped.split())
    if word_count < 2 or word_count > 16:
        return None
    if re.match(r"^(?:page|internal assessment|usn|or)\b", stripped, re.I):
        return None

    line_words = keywords(stripped)
    if not line_words or not rubrics:
        return None

    best_qid = None
    best_score = 0
    line_l = stripped.lower()
    for r in rubrics:
        q_text = r.get("question_text", "")
        q_words = keywords(q_text)
        overlap = len(line_words & q_words)
        q_l = re.sub(r"\s+", " ", q_text.lower())
        if line_l in q_l or q_l[:60] in line_l:
            overlap += 4
        if overlap > best_score:
            best_score = overlap
            best_qid = str(r["question_id"]).upper()

    if best_score >= 2:
        return best_qid
    return None


def normalize_margin_markers(text: str) -> str:
    """Normalize handwritten / OCR-garbled margin markers to QnX) form."""
    # 2] (6) / 2](6) / 2 (6) / 2 [6] / Q2] (6) / 2] 6) / 2] 6] / 36) / 36]  -> Q2B) / Q3B)   (handwritten b read as 6)
    text = re.sub(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)?([1-3])\s*[\]).:-]?\s*[([]?\s*6\s*[)\]:.-]",
        r"\nQ\1B) ",
        text,
        flags=re.IGNORECASE,
    )
    # Ans 1a: / Answer 1(a) / Ans 1(b) / Ans. 1 a / Ans 1-a
    text = re.sub(
        r"(?:^|\n)\s*Ans(?:wer)?\.?\s*(?:No\.?)?\s*([1-3])\s*[([.\-\s]*([a-dA-D])\s*[)\]:.\-]?",
        lambda m: f"\nQ{m.group(1)}{m.group(2).upper()}) ",
        text,
        flags=re.IGNORECASE,
    )
    # Q1 (a) / Q1(a) / Question 1(a) / 1 (a) / 1(a) / 1] a / 3] a / 3) a / 1. a / 1 a) / 3b) / Q1] a] / 1] a]
    text = re.sub(
        r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*)?([1-3])\s*[\]).:-]?\s*[([]?\s*([a-dA-D])(?:\s*[)\]:.-]|\s+)",
        lambda m: f"\nQ{m.group(1)}{m.group(2).upper()}) ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _best_unlabeled_qid(block: str, rubrics: List[Dict], used: List[str]) -> Optional[str]:
    if not rubrics or not block.strip():
        return None
    words = keywords(block[:500])
    scored: List[Tuple[int, str]] = []
    for r in rubrics:
        qid = str(r["question_id"]).upper()
        overlap = len(words & keywords(r.get("question_text", "")))
        scored.append((overlap, qid))
    scored.sort(key=lambda x: (x[0], 0 if x[1] not in used else 1), reverse=True)
    if scored and scored[0][0] >= 2:
        return scored[0][1]
    unused = [r for r in rubrics if str(r["question_id"]).upper() not in used]
    if unused:
        return str(unused[0]["question_id"]).upper()
    return None


def chunk_student_booklet(raw_markdown: str, rubrics: Optional[List[Dict]] = None) -> Dict[str, str]:
    """
    Hybrid boundary detection & multi-page continuation:
      1. Regex-normalize OCR margin quirks (2] (6) -> Q2B, 36) -> Q3B, 3] a -> Q3A).
      2. Split on explicit Q-markers, orphan a)/b), and standalone main questions.
      3. Treat short standalone headings as semantic anchors against teacher questions.
      4. Handle multi-page continuation markers (contd..., continued) seamlessly without
         breaking or collapsing answers.
    Never dump an entire multi-topic booklet into a fallback Q1A.
    """
    rubrics = rubrics or []
    text = clean_page_markers(raw_markdown)
    text = re.sub(r"^\s*\|?\s*Internal Assessment\s*[-–—]?\s*[I1V]+\s*\n?", "", text, flags=re.IGNORECASE)
    text = normalize_margin_markers(text)

    lines = text.splitlines()
    boundaries: List[Tuple[int, str]] = []
    last_main = None

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        # Check if line is a generic continuation marker: "Contd...", "Continued", "(contd)"
        if re.match(r"^(?:#+\s*)?[([]?\s*(?:contd|continued|continuation)\b", line, re.IGNORECASE):
            # It's an explicit continuation of the current question block, keep going
            continue

        # Standalone main question marker (multi-tier parent): e.g. "Question 1", "Q1", "Q1]", "1]", "[1]", "Ans 1:", "Q.1", "1."
        main_only = re.match(
            r"^(?:#+\s*)?(?:(?:Q(?:uestion)?\.?\s*(?:No\.?)?|Ans(?:wer)?\.?)\s*[([]?\s*([1-3])\s*[)\]:.-]?|"
            r"[([]\s*([1-3])\s*[)\]]|([1-3])\s*[)\]:.-])\s*$",
            line,
            re.IGNORECASE,
        )
        if main_only:
            last_main = main_only.group(1) or main_only.group(2) or main_only.group(3)
            continue

        # Question + Sub-question or Sub-question alone (tier 2: e.g. "a]", "b)", "1a)", "Q1A)", "(6)")
        hm = re.match(
            r"^(?:#+\s*)?(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?"
            r"(?:([1-3])\s*[)\].\-\s]*[(\[]?\s*([a-dA-D6])\s*[)\].:\-\s]|([a-dA-D]|[([]?\s*6\s*[)\]])\s*[).:\]\-\s]?)\s*(.*)$",
            line,
            re.IGNORECASE,
        )
        if hm:
            q_num = hm.group(1)
            raw_sub = hm.group(2) or hm.group(3) or "A"
            sub_let = "B" if raw_sub.strip("()[] ") == "6" else raw_sub.strip("()[] ").upper()
            rest = (hm.group(4) or "").strip()
            if q_num:
                last_main = q_num
                qid = f"Q{q_num}{sub_let}"
            elif last_main:
                qid = f"Q{last_main}{sub_let}"
            else:
                qid = _match_orphan_subquestion(
                    rest or " ".join(lines[idx : idx + 4]),
                    sub_let,
                    rubrics,
                    [b[1] for b in boundaries],
                )
                maybe_n = re.match(r"^Q(\d)", qid)
                if maybe_n:
                    last_main = maybe_n.group(1)
            boundaries.append((idx, qid))
            continue

        semantic = _semantic_heading_qid(line, rubrics)
        if semantic:
            maybe_n = re.match(r"^Q(\d)", semantic)
            if maybe_n:
                last_main = maybe_n.group(1)
            boundaries.append((idx, semantic))

    if not boundaries:
        # No markers: score the whole text against every configured question and
        # split by best unique keyword windows instead of collapsing to Q1A.
        return _split_unlabeled_booklet(text, rubrics)

    chunks: Dict[str, str] = {}
    for i, (line_idx, qid) in enumerate(boundaries):
        start = line_idx
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
        block_lines = lines[start:end]
        # Drop the heading line's marker; keep topical heading words.
        if block_lines:
            block_lines = [re.sub(
                r"^(?:#+\s*)?(?:Q(?:uestion)?\.?\s*(?:No\.?)?\s*|Ans(?:wer)?\.?\s*)?"
                r"(?:[1-3]\s*[)\].\-\s]*[(\[]?\s*[a-dA-D6]\s*[)\].:\-\s]|(?:[a-dA-D]|[([]?\s*6\s*[)\]])\s*[).:\]\-\s]?)\s*",
                "",
                block_lines[0],
                flags=re.IGNORECASE,
            )] + block_lines[1:]
        content = "\n".join(block_lines).strip()
        if not content:
            continue
        if qid in chunks:
            chunks[qid] += f"\n\n{content}"
        else:
            chunks[qid] = content

    # Preamble before the first boundary: attach via keywords, not Q1A by default.
    first_idx = boundaries[0][0]
    preamble = "\n".join(lines[:first_idx]).strip()
    if preamble and len(preamble) > 40:
        pqid = _best_unlabeled_qid(preamble, rubrics, list(chunks.keys()))
        if pqid:
            chunks[pqid] = (preamble + "\n\n" + chunks.get(pqid, "")).strip()

    return chunks


def _split_unlabeled_booklet(text: str, rubrics: List[Dict]) -> Dict[str, str]:
    if not text.strip():
        return {}
    if not rubrics:
        return {}

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    assigned: Dict[str, List[str]] = {}
    for para in paragraphs:
        qid = _best_unlabeled_qid(para, rubrics, [])
        if not qid:
            continue
        assigned.setdefault(qid, []).append(para)

    chunks = {qid: "\n\n".join(parts) for qid, parts in assigned.items()}
    if chunks:
        return chunks

    qid = _best_unlabeled_qid(text, rubrics, []) or str(rubrics[0]["question_id"]).upper()
    return {qid: text.strip()}


def calculate_ia_total(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """
    SDMCET IA Rule:
      IA Total = max(Q1A + Q1B, Q2A + Q2B) + (Q3A + Q3B)   out of 20 Marks.
    """
    def g(*keys):
        return sum(float(scores_dict.get(k, 0.0) or 0.0) for k in keys)

    q1_total = g("Q1A", "Q1B")
    q2_total = g("Q2A", "Q2B")
    q3_total = g("Q3A", "Q3B")
    best_choice = max(q1_total, q2_total)
    chosen = "Q1" if q1_total >= q2_total else "Q2"
    total = round(best_choice + q3_total, 2)
    return {
        "q1_total": round(q1_total, 2),
        "q2_total": round(q2_total, 2),
        "q3_total": round(q3_total, 2),
        "chosen_pair": chosen,
        "ia_total": total,
        "max_marks": 20,
    }
